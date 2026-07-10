import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import notification_dispatcher


def _active_alert(alert_id="alert-1"):
    return {
        "id": alert_id,
        "severity": "P1",
        "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
        "description": "Worker without helmet",
        "snapshotUrl": None,
    }


@pytest.fixture(autouse=True)
def _reset_escalation_state():
    notification_dispatcher._escalation_sent.clear()
    notification_dispatcher._escalation_exhausted.clear()
    notification_dispatcher._escalation_attempts.clear()
    yield
    notification_dispatcher._escalation_sent.clear()
    notification_dispatcher._escalation_exhausted.clear()
    notification_dispatcher._escalation_attempts.clear()


def _configure_escalation(monkeypatch, steps, send_alert, alerts=None):
    calls = []
    active_alerts = [_active_alert()] if alerts is None else alerts

    def tracked_send(alert, snapshot_path):
        calls.append((alert, snapshot_path))
        return send_alert(alert, snapshot_path)

    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {"alert_routing": {"escalation_steps": steps}},
    )
    monkeypatch.setattr(
        notification_dispatcher,
        "_CHANNEL_HANDLERS",
        {"telegram": SimpleNamespace(send_alert=tracked_send)},
    )
    monkeypatch.setitem(
        sys.modules,
        "alert_store",
        SimpleNamespace(get_alerts=lambda **_kwargs: active_alerts),
    )
    return calls


def test_failed_escalation_is_retried_before_later_steps(monkeypatch):
    outcomes = iter([False, True, True])
    steps = [
        {"id": 1, "afterMinutes": 1, "role": "Manager", "channel": "telegram"},
        {"id": 2, "afterMinutes": 5, "role": "Director", "channel": "telegram"},
    ]
    calls = _configure_escalation(monkeypatch, steps, lambda *_args: next(outcomes))

    notification_dispatcher._check_escalation()

    assert len(calls) == 1
    assert notification_dispatcher._escalation_sent == {}
    assert notification_dispatcher._escalation_attempts == {("alert-1", 1): 1}

    notification_dispatcher._check_escalation()

    assert len(calls) == 3
    assert notification_dispatcher._escalation_sent == {"alert-1": {1, 2}}
    assert notification_dispatcher._escalation_attempts == {}

    notification_dispatcher._check_escalation()

    assert len(calls) == 3


def test_escalation_runs_chronologically_independent_of_step_ids(monkeypatch):
    steps = [
        {"id": 10, "afterMinutes": 5, "role": "Director", "channel": "telegram"},
        {"id": 20, "afterMinutes": 1, "role": "Manager", "channel": "telegram"},
    ]
    calls = _configure_escalation(monkeypatch, steps, lambda *_args: True)

    notification_dispatcher._check_escalation()

    assert [call[0]["description"].split("]", 1)[0] for call in calls] == [
        "[ESCALATED to Manager after 1min",
        "[ESCALATED to Director after 5min",
    ]
    assert notification_dispatcher._escalation_sent == {"alert-1": {10, 20}}


def test_escalation_normalizes_channel_casing(monkeypatch):
    sent = []
    monkeypatch.setattr(notification_dispatcher, "get_config", lambda: {"alert_outputs": []})
    monkeypatch.setattr(
        notification_dispatcher,
        "_CHANNEL_HANDLERS",
        {"telegram": SimpleNamespace(send_alert=lambda alert, snapshot: sent.append((alert, snapshot)) or True)},
    )

    outcome = notification_dispatcher._send_escalation(
        _active_alert(),
        {"id": 1, "afterMinutes": 1, "role": "Manager", "channel": "TeLeGrAm"},
    )

    assert outcome == notification_dispatcher.ESCALATION_DELIVERED
    assert len(sent) == 1


def test_terminal_unsupported_step_does_not_block_later_step(monkeypatch):
    steps = [
        {"id": 1, "afterMinutes": 1, "role": "Manager", "channel": "SMS"},
        {"id": 2, "afterMinutes": 2, "role": "Director", "channel": "Telegram"},
    ]
    calls = _configure_escalation(monkeypatch, steps, lambda *_args: True)

    notification_dispatcher._check_escalation()

    assert len(calls) == 1
    assert notification_dispatcher._escalation_exhausted == {"alert-1": {1}}
    assert notification_dispatcher._escalation_sent == {"alert-1": {2}}
    assert notification_dispatcher._escalation_attempts == {}


def test_retry_budget_exhaustion_allows_later_step(monkeypatch):
    steps = [
        {"id": 1, "afterMinutes": 1, "role": "Manager", "channel": "telegram"},
        {"id": 2, "afterMinutes": 2, "role": "Director", "channel": "telegram"},
    ]

    def deliver_by_role(alert, _snapshot_path):
        return "Director" in alert["description"]

    calls = _configure_escalation(monkeypatch, steps, deliver_by_role)

    for _ in range(notification_dispatcher.ESCALATION_MAX_ATTEMPTS):
        notification_dispatcher._check_escalation()

    assert sum("Manager" in call[0]["description"] for call in calls) == 3
    assert sum("Director" in call[0]["description"] for call in calls) == 1
    assert notification_dispatcher._escalation_exhausted == {"alert-1": {1}}
    assert notification_dispatcher._escalation_sent == {"alert-1": {2}}
    assert notification_dispatcher._escalation_attempts == {}


def test_escalation_without_a_handler_is_terminal(monkeypatch):
    monkeypatch.setattr(notification_dispatcher, "_CHANNEL_HANDLERS", {})

    outcome = notification_dispatcher._send_escalation(
        _active_alert(),
        {"id": 1, "afterMinutes": 1, "role": "Manager", "channel": "telegram"},
    )

    assert outcome == notification_dispatcher.ESCALATION_TERMINAL


def test_clear_escalation_removes_all_state_for_one_alert():
    notification_dispatcher._escalation_sent.update({"alert-1": {1}, "alert-2": {2}})
    notification_dispatcher._escalation_exhausted.update({"alert-1": {3}, "alert-2": {4}})
    notification_dispatcher._escalation_attempts.update({("alert-1", 5): 2, ("alert-2", 6): 1})

    notification_dispatcher.clear_escalation("alert-1")

    assert notification_dispatcher._escalation_sent == {"alert-2": {2}}
    assert notification_dispatcher._escalation_exhausted == {"alert-2": {4}}
    assert notification_dispatcher._escalation_attempts == {("alert-2", 6): 1}


def test_check_cleans_state_for_alerts_that_are_no_longer_active(monkeypatch):
    notification_dispatcher._escalation_sent["resolved"] = {1}
    notification_dispatcher._escalation_exhausted["resolved"] = {2}
    notification_dispatcher._escalation_attempts[("resolved", 3)] = 1
    steps = [{"id": 1, "afterMinutes": 1, "role": "Manager", "channel": "telegram"}]
    _configure_escalation(monkeypatch, steps, lambda *_args: True, alerts=[])

    notification_dispatcher._check_escalation()

    assert notification_dispatcher._escalation_sent == {}
    assert notification_dispatcher._escalation_exhausted == {}
    assert notification_dispatcher._escalation_attempts == {}


def test_disabling_escalation_clears_process_state(monkeypatch):
    notification_dispatcher._escalation_sent["alert-1"] = {1}
    notification_dispatcher._escalation_exhausted["alert-1"] = {2}
    notification_dispatcher._escalation_attempts[("alert-1", 3)] = 1
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {"alert_routing": {"escalation_steps": []}},
    )

    notification_dispatcher._check_escalation()

    assert notification_dispatcher._escalation_sent == {}
    assert notification_dispatcher._escalation_exhausted == {}
    assert notification_dispatcher._escalation_attempts == {}


def test_escalation_paginates_all_active_alerts(monkeypatch):
    offsets = []
    pages = {
        0: [_active_alert(f"alert-{index}") for index in range(200)],
        200: [_active_alert("alert-200")],
    }

    def get_alerts(**kwargs):
        offsets.append(kwargs["offset"])
        return pages[kwargs["offset"]]

    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {
            "alert_routing": {
                "escalation_steps": [
                    {"id": 1, "afterMinutes": 60, "role": "Manager", "channel": "telegram"},
                ],
            },
        },
    )
    monkeypatch.setitem(sys.modules, "alert_store", SimpleNamespace(get_alerts=get_alerts))
    monkeypatch.setattr(notification_dispatcher, "_CHANNEL_HANDLERS", {})

    notification_dispatcher._check_escalation()

    assert offsets == [0, 200]


def test_notify_with_results_normalizes_requested_channel_casing(monkeypatch):
    calls = []
    monkeypatch.setattr(notification_dispatcher, "get_config", lambda: {})
    monkeypatch.setattr(
        notification_dispatcher,
        "_CHANNEL_HANDLERS",
        {
            "telegram": SimpleNamespace(send_alert=lambda *_args: calls.append("telegram") or True),
            "email": SimpleNamespace(send_alert=lambda *_args: calls.append("email") or False),
        },
    )

    results = notification_dispatcher.notify_with_results(
        {"severity": "P1"},
        channels=["Telegram", "telegram", "EMAIL", "SMS", "inApp"],
    )

    assert calls == ["telegram", "email"]
    assert [(result["channel"], result["success"], result["status"]) for result in results] == [
        ("telegram", True, "delivered"),
        ("email", False, "failed"),
        ("sms", False, "skipped"),
        ("inApp", False, "skipped"),
    ]


def test_notify_reports_actual_external_delivery(monkeypatch):
    outcomes = {
        "telegram": SimpleNamespace(send_alert=lambda *_args: False),
        "email": SimpleNamespace(send_alert=lambda *_args: True),
    }
    monkeypatch.setattr(notification_dispatcher, "_CHANNEL_HANDLERS", outcomes)
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {
            "alert_routing": {
                "channel_matrix": {
                    "P1": {"telegram": True, "email": True},
                },
            },
        },
    )

    results = notification_dispatcher.notify_with_results({"severity": "P1"})

    assert any(result["success"] for result in results)
