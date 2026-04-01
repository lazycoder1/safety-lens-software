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


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_disabled_does_not_send(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": False, "bot_token": "tok", "chat_id": "123", "severities": ["P1"]}}
    telegram_notifier.send_alert(_make_alert())
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

    # Create a temp snapshot file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8fake_jpeg")
        snap_path = f.name

    telegram_notifier.send_alert(_make_alert(), snapshot_path=snap_path)
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

    telegram_notifier.send_alert(_make_alert())
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "sendMessage" in call_kwargs[0][0]
    assert call_kwargs[1]["json"]["chat_id"] == "456"


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_correct_caption_format(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": True, "bot_token": "tok", "chat_id": "123", "severities": ["P1"]}}

    alert = _make_alert()
    telegram_notifier.send_alert(alert)

    call_kwargs = mock_post.call_args
    text = call_kwargs[1]["json"]["text"]
    assert "*P1*" in text
    assert "No Helmet" in text
    assert "Welding Bay" in text
    assert "Zone A" in text


# ── test_connection ──────────────────────────────────────────────────────────

@mock.patch("telegram_notifier.requests.post")
def test_test_connection_success(mock_post):
    import telegram_notifier
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    result = telegram_notifier.test_connection("tok123", "chat456")
    assert result["ok"] is True
    mock_post.assert_called_once()
    assert "sendMessage" in mock_post.call_args[0][0]


@mock.patch("telegram_notifier.requests.post")
def test_test_connection_failure(mock_post):
    import telegram_notifier
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"description": "Unauthorized"}
    mock_post.return_value = mock_resp

    result = telegram_notifier.test_connection("bad_tok", "chat456")
    assert result["ok"] is False
    assert "Unauthorized" in result["error"]


@mock.patch("telegram_notifier.requests.post", side_effect=ConnectionError("no network"))
def test_test_connection_exception(mock_post):
    import telegram_notifier
    result = telegram_notifier.test_connection("tok", "chat")
    assert result["ok"] is False
    assert "no network" in result["error"]
