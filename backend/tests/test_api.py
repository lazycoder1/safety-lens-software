"""Tests for FastAPI API endpoints — uses TestClient with Postgres test DB."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# Set DATABASE_URL to test DB before importing alert_store
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/safetylens_test")
os.environ["DATABASE_URL"] = TEST_DB_URL

_tmpdir = tempfile.mkdtemp()
_test_snapshots = Path(_tmpdir) / "snapshots"
_test_config = Path(_tmpdir) / "test_config.json"

import alert_store
import alert_delivery_store
import audit_store
alert_store.SNAPSHOTS_DIR = _test_snapshots

import auth_store
import config_manager
import notification_dispatcher
config_manager.CONFIG_PATH = _test_config

# Mock YOLO so server doesn't try to load real models
with mock.patch("state.load_model"):
    import server
    import model_manager
    import state

# Use TestClient (no real model loading)
state.model = None
state.yoloe_model = None


@pytest.fixture(autouse=True)
def fresh_state():
    """Reset DB, config, and server state before each test."""
    alert_store.SNAPSHOTS_DIR = _test_snapshots
    alert_store.init_db()

    # Truncate alerts table
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE alert_delivery_outbox, alerts")
        conn.commit()

    # Re-create snapshots dir
    _test_snapshots.mkdir(parents=True, exist_ok=True)

    # Reset config_manager
    config_manager._config = None
    config_manager.CONFIG_PATH = _test_config
    if _test_config.exists():
        _test_config.unlink()
    config_manager.load_config()

    # Clear server state
    state.camera_threads.clear()
    state.vlm_threads.clear()
    state.camera_frames.clear()
    state.camera_detections.clear()
    state.alert_subscribers.clear()
    state.camera_runtime_status.clear()
    model_manager._INSTALL_JOBS.clear()
    model_manager._ACTIVE_JOB_ID = None
    with model_manager._MODEL_LOCK:
        for model_key in model_manager.MODEL_DEFINITIONS:
            model_manager._set_model_state(model_key, status="ready", error=None, active_path=config_manager.CONFIG_PATH, job_id=None)

    yield

    # Clean up snapshot files
    if _test_snapshots.exists():
        for f in _test_snapshots.iterdir():
            f.unlink(missing_ok=True)


client = TestClient(server.app, raise_server_exceptions=False)


def _admin_headers(extra: dict | None = None) -> dict[str, str]:
    token = auth_store.create_token("admin-test", "admin", "admin")
    headers = {"Authorization": f"Bearer {token}"}
    if extra:
        headers.update(extra)
    return headers


def api_get(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.get(path, headers=headers, **kwargs)


def api_post(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.post(path, headers=headers, **kwargs)


def api_put(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.put(path, headers=headers, **kwargs)


def api_delete(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.delete(path, headers=headers, **kwargs)


def _first_camera_id() -> str:
    return next(iter(config_manager.get_config()["cameras"]))


def _create_test_alert(**kwargs):
    defaults = dict(
        camera_id="cam1", camera_name="Test", zone="ZoneA",
        rule="Helmet", severity="P2", confidence=0.9,
        description="Test alert", source="YOLO",
    )
    defaults.update(kwargs)
    return alert_store.create_alert(**defaults)


# ── GET /api/health ──────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in {"ok", "degraded"}
    assert "cameras" in data
    assert all(
        {"sequence", "subscribers", "frameAvailable", "frameAgeSeconds"}
        <= set(camera["stream"])
        for camera in data["cameras"]
    )
    assert {
        "subscribers",
        "subscriberLimitPerCamera",
        "subscriberLimitTotal",
        "rejectedSubscribers",
        "rejectedByCamera",
    } <= set(data["streamFanout"])


# ── GET /api/cameras ─────────────────────────────────────────────────────────

def test_get_cameras():
    resp = api_get("/api/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert isinstance(cameras, list)
    cfg = config_manager.get_config()
    assert len(cameras) == len(cfg["cameras"])
    cam_ids = {c["id"] for c in cameras}
    assert set(cfg["cameras"]) == cam_ids


def test_get_cameras_includes_fields():
    resp = api_get("/api/cameras")
    cam = resp.json()[0]
    assert "name" in cam
    assert "zone" in cam
    assert "zones" in cam
    assert "demo" in cam
    assert "rules" in cam
    assert "enabled" in cam
    assert "status" in cam


def test_get_cameras_includes_saved_zones():
    cfg = config_manager.get_config()
    cam_id = _first_camera_id()
    cfg["cameras"][cam_id]["zones"] = [
        {
            "id": "z1",
            "name": "Saved Zone",
            "type": "restricted",
            "color": "#dc2626",
            "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
        }
    ]
    config_manager.save_config(cfg)

    resp = api_get("/api/cameras")
    cam = next(item for item in resp.json() if item["id"] == cam_id)
    assert cam["zones"][0]["name"] == "Saved Zone"


# ── GET /api/alerts ──────────────────────────────────────────────────────────

def test_get_alerts_empty():
    resp = api_get("/api/alerts")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_alerts_with_data():
    _create_test_alert()
    _create_test_alert(severity="P1")
    resp = api_get("/api/alerts")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_alerts_filter_severity():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    _create_test_alert(severity="P2")
    resp = api_get("/api/alerts?severity=P2")
    assert len(resp.json()) == 2


def test_get_alerts_filter_status():
    a = _create_test_alert()
    _create_test_alert()
    alert_store.acknowledge_alert(a["id"])
    resp = api_get("/api/alerts?status=acknowledged")
    assert len(resp.json()) == 1


def test_get_alerts_filter_camera():
    _create_test_alert(camera_id="cam1")
    _create_test_alert(camera_id="cam2")
    resp = api_get("/api/alerts?cameraId=cam1")
    assert len(resp.json()) == 1


def test_get_alerts_limit():
    for _ in range(5):
        _create_test_alert()
    resp = api_get("/api/alerts?limit=3")
    assert len(resp.json()) == 3


# ── GET /api/alerts/stats ────────────────────────────────────────────────────

def test_alerts_stats_empty():
    resp = api_get("/api/alerts/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["active"] == 0


def test_alerts_stats_with_data():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    resp = api_get("/api/alerts/stats")
    data = resp.json()
    assert data["total"] == 2
    assert data["active"] == 2
    assert data["bySeverity"]["P1"] == 1
    assert data["bySeverity"]["P2"] == 1


# ── GET /api/alerts/time-series ──────────────────────────────────────────────

def test_alert_time_series_endpoint():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    resp = api_get("/api/alerts/time-series")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    bucket = data[0]
    assert "hour" in bucket
    assert "P1" in bucket
    assert "P2" in bucket


def test_alert_time_series_custom_hours():
    resp = api_get("/api/alerts/time-series?hours=1")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── PUT /api/alerts/{id}/acknowledge ─────────────────────────────────────────

def test_acknowledge_alert():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/acknowledge")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "acknowledged"
    assert data["acknowledgedBy"] == "admin"


def test_acknowledge_not_found():
    resp = api_put("/api/alerts/fake-id/acknowledge")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/resolve ─────────────────────────────────────────────

def test_resolve_alert():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/resolve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"


def test_resolve_not_found():
    resp = api_put("/api/alerts/fake-id/resolve")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/snooze ──────────────────────────────────────────────

def test_snooze_alert_default():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/snooze")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "snoozed"
    assert data["snoozedUntil"] is not None


def test_snooze_alert_custom_minutes():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/snooze?minutes=60")
    assert resp.status_code == 200
    assert resp.json()["status"] == "snoozed"


def test_snooze_not_found():
    resp = api_put("/api/alerts/fake-id/snooze")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/false-positive ──────────────────────────────────────

def test_false_positive():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/false-positive")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resolved"
    assert data["falsePositive"] is True


def test_false_positive_not_found():
    resp = api_put("/api/alerts/fake-id/false-positive")
    assert resp.status_code == 404


def test_admin_can_explicitly_replay_recent_ambiguous_terminal_delivery():
    alert = _create_test_alert(
        delivery_targets=[
            {
                "kind": "initial",
                "channel": "webhook",
                "target_key": "initial:webhook:test",
                "context": {},
                "priority": 1,
                "delay_seconds": 0,
            }
        ]
    )
    claimed = alert_delivery_store.claim_due(worker_id="test", lease_seconds=30)
    assert alert_delivery_store.mark_terminal(
        str(claimed["id"]),
        str(claimed["lease_token"]),
        error_code="read_timeout",
        error_message="acceptance unknown",
        acceptance_unknown=True,
    )
    delivery_id = str(claimed["id"])

    refused = api_post(
        f"/api/alert-deliveries/{delivery_id}/replay",
        json={"allowAmbiguous": False},
    )
    assert refused.status_code == 409

    with mock.patch("routers.alerts.alert_delivery_worker.wake") as wake:
        replayed = api_post(
            f"/api/alert-deliveries/{delivery_id}/replay",
            json={"allowAmbiguous": True},
        )
    assert replayed.status_code == 200
    assert replayed.json()["state"] == "pending"
    wake.assert_called_once()
    event = next(
        item
        for item in audit_store.get_recent(limit=20)
        if item["action"] == "alert.delivery_replay"
    )
    assert event["targetId"] == delivery_id
    assert event["details"]["everAcceptanceUnknown"] is True


def test_admin_can_discover_terminal_delivery_ids_without_destination_secrets():
    alert = _create_test_alert(
        delivery_targets=[
            {
                "kind": "initial",
                "channel": "webhook",
                "target_key": "initial:webhook:secret-destination",
                "context": {"url": "https://secret.example/hook", "token": "must-not-leak"},
                "priority": 1,
                "delay_seconds": 0,
            }
        ]
    )
    claimed = alert_delivery_store.claim_due(worker_id="test", lease_seconds=30)
    assert alert_delivery_store.mark_terminal(
        str(claimed["id"]),
        str(claimed["lease_token"]),
        error_code="invalid_configuration",
        error_message="secret provider detail",
    )

    response = api_get(f"/api/alert-deliveries?state=terminal&alertId={alert['id']}")

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["deliveryId"] == str(claimed["id"])
    assert rows[0]["alertId"] == alert["id"]
    assert rows[0]["state"] == "terminal"
    assert rows[0]["channel"] == "webhook"
    serialized = json.dumps(rows)
    assert "secret-destination" not in serialized
    assert "secret.example" not in serialized
    assert "must-not-leak" not in serialized
    assert "secret provider detail" not in serialized


def test_replay_audit_failure_leaves_delivery_terminal():
    alert = _create_test_alert(
        delivery_targets=[
            {
                "kind": "initial",
                "channel": "webhook",
                "target_key": "initial:webhook:test",
                "context": {},
                "priority": 1,
                "delay_seconds": 0,
            }
        ]
    )
    claimed = alert_delivery_store.claim_due(worker_id="test", lease_seconds=30)
    assert alert_delivery_store.mark_terminal(
        str(claimed["id"]),
        str(claimed["lease_token"]),
        error_code="invalid_configuration",
        error_message="fix config first",
    )
    delivery_id = str(claimed["id"])

    with mock.patch(
        "routers.alerts.audit_store.log_event",
        side_effect=RuntimeError("audit database unavailable"),
    ):
        response = api_post(
            f"/api/alert-deliveries/{delivery_id}/replay",
            json={"allowAmbiguous": False},
        )

    assert response.status_code == 500
    assert alert_delivery_store.get_delivery(delivery_id)["state"] == "terminal"


# ── GET /api/snapshots/{filename} ────────────────────────────────────────────

def test_serve_snapshot():
    fake_jpeg = b"\xff\xd8\xff\xe0test"
    a = alert_store.create_alert(
        "cam1", "C", "Z", "R", "P2", 0.9, snapshot_jpeg=fake_jpeg,
    )
    filename = a["snapshotUrl"].split("/")[-1]
    resp = client.get(f"/api/snapshots/{filename}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == fake_jpeg


def test_serve_snapshot_not_found():
    resp = client.get("/api/snapshots/nonexistent.jpg")
    assert resp.status_code == 404


# ── GET /api/config ──────────────────────────────────────────────────────────

def test_get_config():
    resp = api_get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "global" in data
    assert "vlm" in data
    assert "cameras" in data


def test_get_config_never_returns_runtime_secrets():
    cfg = config_manager.get_config()
    cfg["auth"] = {"jwt_secret": "jwt-secret-sentinel"}
    cfg["database"] = {"url": "postgresql://user:db-secret-sentinel@db:5432/safetylens"}
    cfg["telegram"].update({"bot_token": "telegram-secret-sentinel"})
    cfg["email"].update({"smtp_pass": "smtp-secret-sentinel"})
    cfg["webhook"].update({
        "url": "https://hooks.example/secret-sentinel",
        "headers": {
            "Authorization": "Bearer header-secret-sentinel",
            "X-Custom-Key": "custom-header-secret-sentinel",
        },
    })
    config_manager.save_config(cfg)

    token = auth_store.create_token("viewer-test", "viewer", "viewer")
    resp = client.get("/api/config", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    payload = json.dumps(resp.json())
    for sentinel in (
        "jwt-secret-sentinel",
        "db-secret-sentinel",
        "telegram-secret-sentinel",
        "smtp-secret-sentinel",
        "secret-sentinel",
        "header-secret-sentinel",
        "custom-header-secret-sentinel",
    ):
        assert sentinel not in payload
    assert resp.json()["telegram"]["bot_token"] == "***redacted***"
    assert resp.json()["email"]["smtp_pass"] == "***redacted***"
    assert resp.json()["webhook"]["url"] == "***redacted***"
    assert set(resp.json()["webhook"]["headers"].values()) == {"***redacted***"}


# ── PUT /api/config/global ───────────────────────────────────────────────────

@mock.patch("routers.config.restart_all_cameras")
def test_update_global_config(mock_restart):
    resp = api_put("/api/config/global", json={"target_fps": 10})
    assert resp.status_code == 200
    assert resp.json()["target_fps"] == 10
    mock_restart.assert_called_once()


@mock.patch("routers.config.restart_all_cameras")
def test_update_global_partial(mock_restart):
    resp = api_put("/api/config/global", json={"yolo_conf": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["yolo_conf"] == 0.5
    assert data["target_fps"] == 6


# ── PUT /api/config/vlm ─────────────────────────────────────────────────────

def test_update_vlm_config():
    resp = api_put("/api/config/vlm", json={"model": "qwen3.5:35b", "interval": 60})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen3.5:35b"
    assert data["interval"] == 60


@pytest.mark.parametrize(
    "payload",
    [
        {"interval": 0},
        {"temperature": -0.1},
        {"max_tokens": 1000000},
    ],
)
def test_vlm_config_rejects_resource_abusive_values(payload):
    before = json.loads(json.dumps(config_manager.get_config()["vlm"]))

    response = api_put("/api/config/vlm", json=payload)

    assert response.status_code == 422
    assert config_manager.get_config()["vlm"] == before


# ── Telegram config endpoints ────────────────────────────────────────────────

def test_telegram_config_endpoint():
    resp = api_put("/api/config/telegram", json={"enabled": True, "bot_token": "tok123", "chat_id": "456"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["bot_token"] == "***redacted***"
    assert data["chat_id"] == "456"


@pytest.mark.parametrize(
    ("endpoint", "payload", "section"),
    [
        (
            "/api/config/telegram",
            {"enabled": True, "bot_token": "token-without-destination"},
            "telegram",
        ),
        (
            "/api/config/email",
            {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "from_address": "alerts@example.com",
            },
            "email",
        ),
        ("/api/config/webhook", {"enabled": True}, "webhook"),
    ],
)
def test_notification_provider_rejects_incomplete_enabled_config_without_mutating_cache(
    endpoint,
    payload,
    section,
):
    before = json.loads(json.dumps(config_manager.get_config()[section]))

    resp = api_put(endpoint, json=payload)

    assert resp.status_code == 422
    assert "incomplete configuration" in resp.json()["detail"]
    assert config_manager.get_config()[section] == before


def test_notification_provider_partial_update_validates_merged_config():
    assert api_put(
        "/api/config/telegram",
        json={"enabled": True, "bot_token": "tok123", "chat_id": "456"},
    ).status_code == 200

    assert api_put(
        "/api/config/telegram",
        json={"severities": ["P1", "P2", "P3"]},
    ).status_code == 200

    rejected = api_put("/api/config/telegram", json={"chat_id": "   "})
    assert rejected.status_code == 422
    stored = config_manager.get_config()["telegram"]
    assert stored["bot_token"] == "tok123"
    assert stored["chat_id"] == "456"
    assert stored["severities"] == ["P1", "P2", "P3"]


def test_email_rejects_half_configured_authentication_pair():
    resp = api_put(
        "/api/config/email",
        json={
            "enabled": True,
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "mailer",
            "from_address": "alerts@example.com",
            "to_addresses": ["safety@example.com"],
        },
    )

    assert resp.status_code == 422
    assert "smtp_user/smtp_pass" in resp.json()["detail"]


def test_redacted_notification_credentials_are_preserved_on_update():
    cfg = config_manager.get_config()
    cfg["telegram"].update(
        {
            "bot_token": "original-telegram-token",
            "chat_id": "original-chat",
        }
    )
    cfg["email"].update(
        {
            "smtp_host": "smtp.example.com",
            "smtp_user": "original-smtp-user",
            "smtp_pass": "original-smtp-password",
            "from_address": "alerts@example.com",
            "to_addresses": ["safety@example.com"],
        }
    )
    cfg["webhook"].update({
        "url": "https://hooks.example/original",
        "headers": {"Authorization": "Bearer original", "X-Key": "original-key"},
    })
    config_manager.save_config(cfg)

    assert api_put(
        "/api/config/telegram",
        json={"enabled": True, "bot_token": "***redacted***"},
    ).status_code == 200
    assert api_put(
        "/api/config/email",
        json={"enabled": True, "smtp_pass": "***redacted***"},
    ).status_code == 200
    assert api_put(
        "/api/config/webhook",
        json={
            "enabled": True,
            "url": "***redacted***",
            "headers": {
                "Authorization": "***redacted***",
                "X-Key": "***redacted***",
            },
        },
    ).status_code == 200

    stored = config_manager.get_config()
    assert stored["telegram"]["bot_token"] == "original-telegram-token"
    assert stored["email"]["smtp_pass"] == "original-smtp-password"
    assert stored["webhook"]["url"] == "https://hooks.example/original"
    assert stored["webhook"]["headers"] == {
        "Authorization": "Bearer original",
        "X-Key": "original-key",
    }
    new_webhook_url = "https://hooks.example/services/new-secret-path"
    assert api_put(
        "/api/config/webhook",
        json={"url": new_webhook_url},
    ).status_code == 200
    webhook_audit = next(
        event
        for event in audit_store.get_recent(limit=50)
        if event["action"] == "config.webhook_update"
    )
    assert new_webhook_url not in json.dumps(webhook_audit)
    assert webhook_audit["details"]["updates"]["url"] == "***redacted***"
    webhook_audits = [
        event
        for event in audit_store.get_recent(limit=50)
        if event["action"] == "config.webhook_update"
    ]
    serialized_audits = json.dumps(webhook_audits)
    assert "Bearer original" not in serialized_audits
    assert "original-key" not in serialized_audits


@mock.patch("telegram_notifier.test_connection")
def test_telegram_test_endpoint(mock_test):
    mock_test.return_value = {"ok": True}
    resp = api_post("/api/config/telegram/test")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ── Alert routing config endpoints ───────────────────────────────────────────

def test_alert_routing_normalizes_supported_escalation_channels():
    assert api_put(
        "/api/config/telegram",
        json={
            "enabled": True,
            "bot_token": "123456:telegram-token",
            "chat_id": "456",
        },
    ).status_code == 200
    assert api_put(
        "/api/config/email",
        json={
            "enabled": True,
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "from_address": "alerts@example.com",
            "to_addresses": ["safety@example.com"],
        },
    ).status_code == 200
    resp = api_put(
        "/api/config/alert-routing",
        json={
            "escalation_steps": [
                {"id": 7, "afterMinutes": 3, "role": "Manager", "channel": " Telegram "},
                {"id": 3, "afterMinutes": 8, "role": "Director", "channel": "EMAIL"},
            ],
        },
    )

    assert resp.status_code == 200
    assert [step["channel"] for step in resp.json()["escalation_steps"]] == ["telegram", "email"]
    assert all(step["enabled"] is True for step in resp.json()["escalation_steps"])
    assert all(step["severities"] == ["P1", "P2"] for step in resp.json()["escalation_steps"])


def test_alert_routing_rejects_active_escalation_with_disabled_provider_without_saving():
    before = json.loads(
        json.dumps(config_manager.get_config()["alert_routing"]["escalation_steps"])
    )

    resp = api_put(
        "/api/config/alert-routing",
        json={
            "escalation_steps": [
                {
                    "id": 7,
                    "afterMinutes": 3,
                    "role": "Manager",
                    "channel": "telegram",
                }
            ],
        },
    )

    assert resp.status_code == 422
    assert "provider" in resp.json()["detail"]
    assert config_manager.get_config()["alert_routing"]["escalation_steps"] == before


def test_alert_routing_rejects_duplicate_enabled_step_identity_without_saving():
    assert api_put(
        "/api/config/email",
        json={
            "enabled": True,
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "from_address": "alerts@example.com",
            "to_addresses": ["safety@example.com"],
        },
    ).status_code == 200
    before = json.loads(
        json.dumps(config_manager.get_config()["alert_routing"]["escalation_steps"])
    )

    resp = api_put(
        "/api/config/alert-routing",
        json={
            "escalation_steps": [
                {
                    "id": 7,
                    "afterMinutes": 3,
                    "role": "Manager",
                    "channel": "email",
                },
                {
                    "id": "7",
                    "afterMinutes": 10,
                    "role": "Director",
                    "channel": "email",
                },
            ],
        },
    )

    assert resp.status_code == 422
    assert "unique id per channel" in resp.json()["detail"]
    assert config_manager.get_config()["alert_routing"]["escalation_steps"] == before


def test_disabled_escalation_step_does_not_require_or_create_provider_work():
    resp = api_put(
        "/api/config/alert-routing",
        json={
            "escalation_steps": [
                {
                    "id": 7,
                    "enabled": False,
                    "afterMinutes": 3,
                    "role": "Manager",
                    "channel": "telegram",
                }
            ],
        },
    )

    assert resp.status_code == 200
    targets = notification_dispatcher.resolve_delivery_targets(
        config_manager.get_config(),
        {"id": "alert", "severity": "P1"},
    )
    assert targets == []


def test_active_escalation_prevents_later_provider_disable_or_scope_removal():
    assert api_put(
        "/api/config/telegram",
        json={
            "enabled": True,
            "bot_token": "123456:telegram-token",
            "chat_id": "456",
            "severities": ["P1", "P2"],
        },
    ).status_code == 200
    assert api_put(
        "/api/config/alert-routing",
        json={
            "escalation_steps": [
                {
                    "id": 7,
                    "afterMinutes": 3,
                    "role": "Manager",
                    "channel": "telegram",
                    "severities": ["P1", "P2"],
                }
            ],
        },
    ).status_code == 200

    disabled = api_put("/api/config/telegram", json={"enabled": False})
    narrowed = api_put("/api/config/telegram", json={"severities": ["P1"]})

    assert disabled.status_code == 422
    assert narrowed.status_code == 422
    assert "filters a severity required by the step" in narrowed.json()["detail"]
    stored = config_manager.get_config()["telegram"]
    assert stored["enabled"] is True
    assert stored["severities"] == ["P1", "P2"]


def test_alert_routing_rejects_unimplemented_escalation_channel():
    resp = api_put(
        "/api/config/alert-routing",
        json={
            "escalation_steps": [
                {"id": 1, "afterMinutes": 3, "role": "Manager", "channel": "SMS"},
            ],
        },
    )

    assert resp.status_code == 422
    assert "Unsupported escalation channel: sms" in resp.json()["detail"]


def test_alert_routing_rejects_enabled_unimplemented_matrix_channel_without_saving():
    before = json.loads(
        json.dumps(config_manager.get_config()["alert_routing"]["channel_matrix"])
    )

    resp = api_put(
        "/api/config/alert-routing",
        json={"channel_matrix": {"P1": {"inApp": True, "sms": True}}},
    )

    assert resp.status_code == 422
    assert "Unsupported enabled alert-routing channel: sms" in resp.json()["detail"]
    assert config_manager.get_config()["alert_routing"]["channel_matrix"] == before


def test_alert_routing_rejects_provider_that_is_not_ready():
    disabled = api_put(
        "/api/config/alert-routing",
        json={"channel_matrix": {"P1": {"telegram": True}}},
    )
    assert disabled.status_code == 422
    assert "provider is disabled" in disabled.json()["detail"]

    cfg = config_manager.get_config()
    cfg["telegram"].update({"enabled": True, "bot_token": "token", "chat_id": ""})
    config_manager.save_config(cfg)
    incomplete = api_put(
        "/api/config/alert-routing",
        json={"channel_matrix": {"P1": {"telegram": True}}},
    )
    assert incomplete.status_code == 422
    assert "incomplete provider configuration" in incomplete.json()["detail"]


def test_alert_routing_accepts_ready_provider_and_prevents_disabling_it():
    assert api_put(
        "/api/config/telegram",
        json={"enabled": True, "bot_token": "tok123", "chat_id": "456"},
    ).status_code == 200
    routed = api_put(
        "/api/config/alert-routing",
        json={"channel_matrix": {"P1": {"inApp": True, "telegram": True}}},
    )
    assert routed.status_code == 200

    disable = api_put("/api/config/telegram", json={"enabled": False})
    assert disable.status_code == 422
    assert "provider is disabled" in disable.json()["detail"]
    assert config_manager.get_config()["telegram"]["enabled"] is True


def test_alert_routing_rejects_provider_severity_filter_conflict():
    assert api_put(
        "/api/config/webhook",
        json={
            "enabled": True,
            "url": "https://hooks.example/safety",
            "severities": ["P2"],
        },
    ).status_code == 200

    resp = api_put(
        "/api/config/alert-routing",
        json={"channel_matrix": {"P1": {"webhook": True}}},
    )

    assert resp.status_code == 422
    assert "provider filters that severity" in resp.json()["detail"]


@mock.patch("notification_dispatcher.notify_with_results")
def test_alert_routing_test_returns_honest_channel_results(mock_notify):
    mock_notify.return_value = [
        {"channel": "telegram", "success": True, "status": "delivered", "message": "Delivered"},
        {"channel": "email", "success": False, "status": "failed", "message": "SMTP unavailable"},
    ]

    resp = api_post(
        "/api/config/alert-routing/test",
        json={"severity": "P1", "channels": ["Telegram", "email"]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert data["results"] == mock_notify.return_value
    alert, snapshot_path = mock_notify.call_args.args
    assert alert["severity"] == "P1"
    assert snapshot_path is None
    assert mock_notify.call_args.kwargs["channels"] == ["Telegram", "email"]
    assert mock_notify.call_args.kwargs["test_request"] is True


@mock.patch("notification_dispatcher.notify_with_results")
def test_alert_routing_test_expands_default_matrix_as_explicit_test_channels(mock_notify):
    mock_notify.return_value = []

    resp = api_post("/api/config/alert-routing/test", json={"severity": "P1"})

    assert resp.status_code == 200
    assert resp.json()["ok"] is False
    assert mock_notify.call_args.kwargs["channels"] == ["inApp"]
    assert mock_notify.call_args.kwargs["test_request"] is True


# ── GET /api/alert-rules-available ───────────────────────────────────────────

def test_available_alert_rules_endpoint():
    resp = api_get("/api/alert-rules-available")
    assert resp.status_code == 200
    data = resp.json()
    assert "alert_mobile_phone" in data
    assert "alert_animal" in data
    assert "alert_person" in data
    assert "alert_vehicle" in data
    assert data["alert_mobile_phone"]["rule"] == "Mobile Phone Usage"
    assert data["alert_mobile_phone"]["severity"] == "P3"


# ── GET /api/videos ──────────────────────────────────────────────────────────

def test_list_videos(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.avi").touch()
    (tmp_path / "c.mp4").touch()
    (tmp_path / "readme.txt").touch()
    from routers import misc
    misc.VIDEO_DIR = tmp_path
    resp = api_get("/api/videos")
    assert resp.status_code == 200
    videos = resp.json()
    assert "a.mp4" in videos
    assert "b.avi" in videos
    assert "c.mp4" in videos
    assert "readme.txt" not in videos


def test_videos_includes_avi(tmp_path):
    (tmp_path / "test.avi").touch()
    from routers import misc
    misc.VIDEO_DIR = tmp_path
    resp = api_get("/api/videos")
    assert resp.status_code == 200
    assert "test.avi" in resp.json()


# ── Camera CRUD ──────────────────────────────────────────────────────────────

@mock.patch("routers.cameras.start_camera")
def test_add_camera(mock_start):
    resp = api_post("/api/cameras", json={
        "name": "New Cam", "video": "test.mp4", "zone": "ZoneX",
        "demo": "yolo", "rules": ["Test Rule"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "New Cam"
    assert "id" in data
    mock_start.assert_called_once()


@mock.patch("routers.cameras.model_manager.missing_model_keys", return_value=["coco_primary"])
def test_add_camera_returns_missing_models(_mock_missing):
    resp = api_post("/api/cameras", json={
        "name": "Needs Model", "video": "test.mp4", "zone": "ZoneX",
        "profile": "general_safety", "capabilities": ["person_presence"],
    })
    assert resp.status_code == 409
    data = resp.json()
    assert data["code"] == "missing_models"
    assert "coco_primary" in data["missing_model_keys"]


@mock.patch("routers.cameras.restart_camera")
def test_update_camera(mock_restart):
    cam_id = _first_camera_id()
    resp = api_put(f"/api/cameras/{cam_id}", json={"name": "Updated Name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated Name"
    mock_restart.assert_called_once_with(cam_id)


def test_get_cameras_derives_display_rules_from_safety_rule_ids():
    from routers.cameras import _camera_public_payload

    cfg = config_manager.get_config()
    cam_id = next(iter(cfg["cameras"]))
    cfg["cameras"][cam_id]["rules"] = ["Stale Rule Label"]
    cfg["cameras"][cam_id]["safety_rule_ids"] = ["ppe_helmet", "alert_mobile_phone"]
    config_manager.save_config(cfg)

    camera = _camera_public_payload(cam_id, cfg["cameras"][cam_id], cfg)
    assert camera["rules"] == ["Helmet", "Mobile Phone Usage"]


def test_create_safety_rule_with_threshold():
    resp = api_post("/api/safety-rules", json={
        "name": "Phone Near Workstation",
        "type": "alert",
        "model": "yolo",
        "classes": ["cell phone"],
        "severity": "P3",
        "threshold": 4,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold"] == 4


def test_update_safety_rule_threshold():
    resp = api_put("/api/safety-rules/alert_mobile_phone", json={"threshold": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold"] == 3

    cfg = config_manager.get_config()
    rule = next(rule for rule in cfg["safety_rules"] if rule["id"] == "alert_mobile_phone")
    assert rule["threshold"] == 3


def test_update_safety_rule_threshold_can_clear():
    resp = api_put("/api/safety-rules/alert_mobile_phone", json={"threshold": 3})
    assert resp.status_code == 200

    resp = api_put("/api/safety-rules/alert_mobile_phone", json={"threshold": None})
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold"] is None

    cfg = config_manager.get_config()
    rule = next(rule for rule in cfg["safety_rules"] if rule["id"] == "alert_mobile_phone")
    assert rule["threshold"] is None


def test_update_camera_replaces_stale_legacy_detection_fields():
    cfg = config_manager.get_config()
    cam_id = _first_camera_id()
    cfg["cameras"][cam_id].update({
        "safety_rule_ids": ["alert_animal", "alert_mobile_phone", "ppe_helmet", "ppe_hairnet"],
        "ppe_rule_ids": ["ppe_helmet", "ppe_hairnet"],
        "alert_classes": ["animal_intrusion", "mobile_phone"],
        "yoloe_classes": ["person", "hard hat", "hairnet", "cell phone"],
        "capabilities": ["animal_presence", "mobile_phone", "helmet_required", "hairnet_required"],
    })
    config_manager.save_config(cfg)

    resp = api_put(f"/api/cameras/{cam_id}", json={
        "capabilities": ["hairnet_required"],
        "safety_rule_ids": ["ppe_hairnet"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_rule_ids"] == ["ppe_hairnet"]
    assert data["ppe_rule_ids"] == ["ppe_hairnet"]
    assert data["alert_classes"] == []
    assert data["capabilities"] == ["hairnet_required"]


def test_update_camera_can_remove_stale_ppe_rules():
    cfg = config_manager.get_config()
    cam_id = _first_camera_id()
    cfg["cameras"][cam_id].update({
        "safety_rule_ids": ["alert_animal", "ppe_helmet", "ppe_vest", "ppe_hairnet"],
        "ppe_rule_ids": ["ppe_helmet", "ppe_vest", "ppe_hairnet"],
        "capabilities": ["animal_presence", "helmet_required", "vest_required", "hairnet_required"],
        "yoloe_classes": ["person", "hard hat", "safety vest", "hairnet"],
    })
    config_manager.save_config(cfg)

    resp = api_put(f"/api/cameras/{cam_id}", json={
        "capabilities": ["animal_presence"],
        "safety_rule_ids": ["alert_animal"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_rule_ids"] == ["alert_animal"]
    assert data["ppe_rule_ids"] == []
    assert data["capabilities"] == ["animal_presence"]


def test_preview_camera_plan():
    resp = api_post("/api/camera-plans/preview", json={
        "profile": "work_zone_ppe",
        "capabilities": ["helmet_required", "zone_intrusion"],
        "stream_type": "file",
        "video": "test.mp4",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_plan"]["run_coco_primary"] is True
    assert data["execution_plan"]["run_ppe_specialist"] is True


def test_update_camera_not_found():
    resp = api_put("/api/cameras/cam999", json={"name": "X"})
    assert resp.status_code == 404


@mock.patch("routers.cameras.stream_fanout.retire")
@mock.patch("routers.cameras.stop_camera", return_value=True)
def test_delete_camera(mock_stop, mock_retire):
    cam_id = _first_camera_id()
    resp = api_delete(f"/api/cameras/{cam_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == cam_id
    mock_stop.assert_called_once_with(cam_id)
    mock_retire.assert_called_once_with(cam_id)

    cfg = config_manager.get_config()
    assert cam_id not in cfg["cameras"]


@mock.patch("routers.cameras.stream_fanout.retire")
@mock.patch("routers.cameras.stop_camera", return_value=False)
def test_delete_camera_waits_for_stuck_worker(mock_stop, mock_retire):
    cam_id = _first_camera_id()

    resp = api_delete(f"/api/cameras/{cam_id}")

    assert resp.status_code == 409
    assert resp.json()["detail"] == "Camera worker is still stopping; retry deletion"
    mock_stop.assert_called_once_with(cam_id)
    mock_retire.assert_not_called()
    assert cam_id in config_manager.get_config()["cameras"]


def test_delete_camera_not_found():
    resp = api_delete("/api/cameras/cam999")
    assert resp.status_code == 404


# ── Enhanced stats ──────────────────────────────────────────────────────────

def test_stats_includes_breakdowns():
    alert_store.create_alert("cam1", "Cam A", "Zone 1", "Helmet", "P2", 0.9)
    alert_store.create_alert("cam2", "Cam B", "Zone 2", "Vest", "P1", 0.8)

    resp = api_get("/api/alerts/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["active"] == 2
    assert data["acknowledged"] == 0
    assert data["resolved"] == 0
    assert data["byRule"]["Helmet"] == 1
    assert data["byRule"]["Vest"] == 1
    assert data["byZone"]["Zone 1"] == 1
    assert data["byZone"]["Zone 2"] == 1
    assert data["byCamera"]["Cam A"] == 1
    assert data["byCamera"]["Cam B"] == 1
