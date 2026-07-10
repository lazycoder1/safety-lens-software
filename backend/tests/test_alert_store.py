"""Tests for alert_store module — PostgreSQL alert persistence."""

from datetime import datetime, timedelta, timezone
import os
import tempfile
import threading
from pathlib import Path

import pytest
import psycopg2

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/rakshak_lens_test")

# Set DATABASE_URL before importing alert_store
os.environ["DATABASE_URL"] = TEST_DB_URL

import alert_store

_test_snapshots = Path(tempfile.mkdtemp()) / "snapshots"


@pytest.fixture(autouse=True)
def fresh_db():
    """Truncate alerts table and re-init before each test."""
    alert_store.SNAPSHOTS_DIR = _test_snapshots
    alert_store.init_db()
    # Truncate
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE alert_delivery_outbox, alerts")
        conn.commit()
    # Re-create snapshots dir
    _test_snapshots.mkdir(parents=True, exist_ok=True)
    yield
    # Clean up snapshot files
    if _test_snapshots.exists():
        for f in _test_snapshots.iterdir():
            f.unlink(missing_ok=True)


# ── init_db ──────────────────────────────────────────────────────────────────

def test_init_db_creates_snapshots_dir():
    assert _test_snapshots.exists()
    assert _test_snapshots.is_dir()


def test_init_db_creates_table():
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE tablename = 'alerts'")
            tables = cur.fetchall()
    assert len(tables) == 1


def test_init_db_creates_indexes():
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'alerts' AND indexname LIKE 'idx_alerts_%'"
            )
            rows = cur.fetchall()
    names = {row[0] for row in rows}
    assert "idx_alerts_status" in names
    assert "idx_alerts_severity" in names
    assert "idx_alerts_timestamp" in names
    assert "idx_alerts_camera" in names
    assert "idx_alerts_active_stale" in names


# ── create_alert ─────────────────────────────────────────────────────────────

def test_create_alert_returns_dict_with_expected_keys():
    alert = alert_store.create_alert(
        camera_id="cam1", camera_name="Test Cam", zone="Zone A",
        rule="No Helmet", severity="P2", confidence=0.87,
        description="Worker without helmet", source="YOLO",
    )
    expected_keys = {
        "id", "severity", "status", "rule", "cameraId", "cameraName",
        "zone", "confidence", "timestamp", "source", "description",
        "cleanSnapshotUrl", "bboxes", "deliveryResults", "policyId",
        "priority", "message", "metadata",
        "snapshotUrl", "acknowledgedBy", "acknowledgedAt",
        "resolvedAt", "snoozedUntil", "falsePositive",
    }
    assert set(alert.keys()) == expected_keys
    assert alert["cameraId"] == "cam1"
    assert alert["cameraName"] == "Test Cam"
    assert alert["zone"] == "Zone A"
    assert alert["rule"] == "No Helmet"
    assert alert["severity"] == "P2"
    assert alert["status"] == "active"
    assert alert["source"] == "YOLO"
    assert alert["description"] == "Worker without helmet"
    assert alert["snapshotUrl"] is None
    assert alert["deliveryResults"] == []
    assert alert["policyId"] is None
    assert alert["message"] is None
    assert alert["metadata"] == {}
    assert alert["acknowledgedBy"] is None
    assert alert["falsePositive"] is False


def test_create_alert_id_is_full_uuid():
    alert = alert_store.create_alert(
        camera_id="cam1", camera_name="C", zone="Z",
        rule="R", severity="P3", confidence=0.5,
    )
    assert len(alert["id"]) == 36


def test_create_alert_confidence_rounded():
    alert = alert_store.create_alert(
        camera_id="cam1", camera_name="C", zone="Z",
        rule="R", severity="P3", confidence=0.87654321,
    )
    assert alert["confidence"] == 0.88


def test_create_alert_with_snapshot():
    fake_jpeg = b"\xff\xd8\xff\xe0fake_jpeg_data"
    alert = alert_store.create_alert(
        camera_id="cam1", camera_name="Test", zone="Z",
        rule="Test", severity="P3", confidence=0.5,
        snapshot_jpeg=fake_jpeg,
    )
    assert alert["snapshotUrl"] is not None
    assert alert["snapshotUrl"].startswith("/api/snapshots/")

    # Verify file was written
    filename = alert["snapshotUrl"].split("/")[-1]
    snapshot_file = _test_snapshots / filename
    assert snapshot_file.exists()
    assert snapshot_file.read_bytes() == fake_jpeg


def test_create_alert_generates_unique_ids():
    ids = set()
    for _ in range(20):
        alert = alert_store.create_alert(
            camera_id="cam1", camera_name="C", zone="Z",
            rule="R", severity="P3", confidence=0.5,
        )
        ids.add(alert["id"])
    assert len(ids) == 20


def test_create_alert_persists_to_db():
    alert = alert_store.create_alert(
        camera_id="cam1", camera_name="C", zone="Z",
        rule="R", severity="P3", confidence=0.5,
    )
    fetched = alert_store.get_alert(alert["id"])
    assert fetched is not None
    assert fetched["id"] == alert["id"]
    assert fetched["severity"] == "P3"


# ── get_alerts ───────────────────────────────────────────────────────────────

def _seed_alerts():
    """Seed DB with alerts for filtering tests."""
    alert_store.create_alert("cam1", "Cam1", "ZoneA", "Helmet", "P1", 0.9, source="YOLO")
    alert_store.create_alert("cam1", "Cam1", "ZoneA", "Vest", "P2", 0.8, source="YOLO")
    alert_store.create_alert("cam2", "Cam2", "ZoneB", "Phone", "P3", 0.7, source="YOLO")
    alert_store.create_alert("cam2", "Cam2", "ZoneB", "VLM", "P2", 0.6, source="VLM")


def test_get_alerts_returns_list():
    _seed_alerts()
    alerts = alert_store.get_alerts()
    assert isinstance(alerts, list)
    assert len(alerts) == 4


def test_get_alerts_filter_by_severity():
    _seed_alerts()
    alerts = alert_store.get_alerts(severity="P2")
    assert len(alerts) == 2
    assert all(a["severity"] == "P2" for a in alerts)


def test_get_alerts_filter_by_status():
    _seed_alerts()
    alerts = alert_store.get_alerts(status="active")
    assert len(alerts) == 4

    # Acknowledge one, then filter
    alert_store.acknowledge_alert(alerts[0]["id"])
    active = alert_store.get_alerts(status="active")
    assert len(active) == 3
    acked = alert_store.get_alerts(status="acknowledged")
    assert len(acked) == 1


def test_get_alerts_filter_by_camera_id():
    _seed_alerts()
    alerts = alert_store.get_alerts(camera_id="cam2")
    assert len(alerts) == 2
    assert all(a["cameraId"] == "cam2" for a in alerts)


def test_get_alerts_limit_and_offset():
    _seed_alerts()
    limited = alert_store.get_alerts(limit=2)
    assert len(limited) == 2

    all_alerts = alert_store.get_alerts()
    offset_alerts = alert_store.get_alerts(offset=2)
    assert len(offset_alerts) == 2
    assert offset_alerts[0]["id"] == all_alerts[2]["id"]


# ── get_alert ────────────────────────────────────────────────────────────────

def test_get_alert_found():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P3", 0.5)
    fetched = alert_store.get_alert(alert["id"])
    assert fetched == alert


def test_get_alert_not_found():
    result = alert_store.get_alert("nonexistent")
    assert result is None


# ── acknowledge_alert ────────────────────────────────────────────────────────

def test_acknowledge_alert():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    result = alert_store.acknowledge_alert(alert["id"])
    assert result["status"] == "acknowledged"
    assert result["acknowledgedBy"] == "Admin"
    assert result["acknowledgedAt"] is not None


def test_acknowledge_alert_custom_user():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    result = alert_store.acknowledge_alert(alert["id"], by="Operator1")
    assert result["acknowledgedBy"] == "Operator1"


def test_acknowledge_already_resolved_is_noop():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    alert_store.resolve_alert(alert["id"])
    result = alert_store.acknowledge_alert(alert["id"])
    assert result["status"] == "resolved"


def test_acknowledge_nonexistent():
    result = alert_store.acknowledge_alert("fake-id")
    assert result is None


# ── resolve_alert ────────────────────────────────────────────────────────────

def test_resolve_alert():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    result = alert_store.resolve_alert(alert["id"])
    assert result["status"] == "resolved"
    assert result["resolvedAt"] is not None


def test_resolve_acknowledged_alert():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    alert_store.acknowledge_alert(alert["id"])
    result = alert_store.resolve_alert(alert["id"])
    assert result["status"] == "resolved"


def test_resolve_already_resolved_is_noop():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    alert_store.resolve_alert(alert["id"])
    result = alert_store.resolve_alert(alert["id"])
    assert result["status"] == "resolved"


def test_resolve_nonexistent():
    result = alert_store.resolve_alert("fake-id")
    assert result is None


# ── snooze_alert ─────────────────────────────────────────────────────────────

def test_snooze_alert():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    result = alert_store.snooze_alert(alert["id"], minutes=30)
    assert result["status"] == "snoozed"
    assert result["snoozedUntil"] is not None


def test_snooze_default_minutes():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    result = alert_store.snooze_alert(alert["id"])
    assert result["status"] == "snoozed"


def test_snooze_already_resolved_is_noop():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    alert_store.resolve_alert(alert["id"])
    result = alert_store.snooze_alert(alert["id"])
    assert result["status"] == "resolved"


def test_snooze_nonexistent():
    result = alert_store.snooze_alert("fake-id")
    assert result is None


# ── mark_false_positive ──────────────────────────────────────────────────────

def test_mark_false_positive():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    result = alert_store.mark_false_positive(alert["id"])
    assert result["status"] == "resolved"
    assert result["falsePositive"] is True
    assert result["resolvedAt"] is not None


def test_false_positive_nonexistent():
    result = alert_store.mark_false_positive("fake-id")
    assert result is None


# ── get_stats ────────────────────────────────────────────────────────────────

def test_get_stats():
    stats = alert_store.get_stats()
    assert stats["total"] == 0
    assert stats["active"] == 0
    assert stats["acknowledged"] == 0
    assert stats["resolved"] == 0
    assert stats["bySeverity"] == {}
    assert stats["byRule"] == {}
    assert stats["byZone"] == {}
    assert stats["byCamera"] == {}

    # Add data
    alert_store.create_alert("cam1", "C", "Z", "R", "P1", 0.9)
    alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.8)
    alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.7)

    stats = alert_store.get_stats()
    assert stats["total"] == 3
    assert stats["active"] == 3
    assert stats["bySeverity"]["P1"] == 1
    assert stats["bySeverity"]["P2"] == 2
    assert stats["byRule"]["R"] == 3
    assert stats["byZone"]["Z"] == 3
    assert stats["byCamera"]["C"] == 3


def test_get_stats_resolved_tracking():
    a1 = alert_store.create_alert("cam1", "C", "Z", "R1", "P1", 0.9)
    alert_store.create_alert("cam1", "C", "Z", "R2", "P2", 0.8)
    alert_store.resolve_alert(a1["id"])

    stats = alert_store.get_stats()
    assert stats["total"] == 2
    assert stats["active"] == 1
    assert stats["resolved"] == 1
    assert stats["bySeverity"]["P2"] == 1
    assert "P1" not in stats["bySeverity"]


# ── get_time_series ──────────────────────────────────────────────────────────

def test_get_time_series():
    # Empty case
    ts = alert_store.get_time_series(hours=24)
    assert isinstance(ts, list)
    assert len(ts) == 0

    # Create some alerts (timestamps are "now", so they'll be in the current hour)
    alert_store.create_alert("cam1", "C", "Z", "R", "P1", 0.9)
    alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.8)
    alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.7)

    ts = alert_store.get_time_series(hours=24)
    assert len(ts) >= 1
    # Current hour bucket should have our alerts
    bucket = ts[0]
    assert "hour" in bucket
    assert "P1" in bucket
    assert "P2" in bucket
    assert bucket["P1"] == 1
    assert bucket["P2"] == 2


# ── camelCase output ─────────────────────────────────────────────────────────

def test_output_uses_camel_case():
    alert = alert_store.create_alert("cam1", "C", "Z", "R", "P2", 0.9)
    camel_keys = {
        "id", "severity", "status", "rule", "cameraId", "cameraName",
        "zone", "confidence", "timestamp", "source", "description",
        "cleanSnapshotUrl", "bboxes", "deliveryResults", "policyId",
        "priority", "message", "metadata",
        "snapshotUrl", "acknowledgedBy", "acknowledgedAt",
        "resolvedAt", "snoozedUntil", "falsePositive",
    }
    assert set(alert.keys()) == camel_keys


# ── thread safety ────────────────────────────────────────────────────────────

def test_concurrent_creates():
    """Multiple threads creating alerts should not corrupt data."""
    errors = []

    def worker(n):
        try:
            for i in range(10):
                alert_store.create_alert(
                    f"cam{n}", f"Cam{n}", "Z", "R", "P3", 0.5,
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    alerts = alert_store.get_alerts(limit=500)
    assert len(alerts) == 40


def test_auto_resolve_stale_alerts_uses_bounded_batches_and_cancels_escalations():
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    target = {
        "kind": "escalation",
        "target_key": "email:ops@example.com:step-1",
        "channel": "email",
        "delay_seconds": 30 * 24 * 60 * 60,
    }
    for index in range(3):
        alert_store.create_alert(
            f"cam{index}",
            f"Camera {index}",
            "Zone A",
            "Helmet",
            "P2",
            0.9,
            timestamp=old_timestamp,
            delivery_targets=[target],
        )

    resolved, has_more = alert_store.auto_resolve_stale_alerts_batch(
        batch_size=2
    )

    assert (resolved, has_more) == (2, True)
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT alerts.status, delivery.state, COUNT(*)
                FROM alerts
                JOIN alert_delivery_outbox AS delivery
                  ON delivery.alert_id = alerts.id
                GROUP BY alerts.status, delivery.state
                ORDER BY alerts.status, delivery.state
                """
            )
            states = cur.fetchall()
    assert sorted(states) == [
        ("active", "pending", 1),
        ("resolved", "cancelled", 2),
    ]

    resolved, has_more = alert_store.auto_resolve_stale_alerts_batch(
        batch_size=2
    )

    assert (resolved, has_more) == (1, False)
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status")
            assert cur.fetchall() == [("resolved", 3)]
            cur.execute(
                "SELECT state, COUNT(*) FROM alert_delivery_outbox GROUP BY state"
            )
            assert cur.fetchall() == [("cancelled", 3)]


def test_cleanup_snapshots_prunes_old_resolved_alerts():
    fake_jpeg = b"\xff\xd8\xff\xe0old_jpeg"
    alert = alert_store.create_alert(
        camera_id="cam1",
        camera_name="Test",
        zone="Zone A",
        rule="Helmet",
        severity="P2",
        confidence=0.87,
        snapshot_jpeg=fake_jpeg,
        clean_snapshot_jpeg=fake_jpeg,
    )
    alert_store.resolve_alert(alert["id"])
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()

    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET timestamp = %s, status = 'resolved' WHERE id = %s",
                (old_timestamp, alert["id"]),
            )
        conn.commit()

    result = alert_store.cleanup_snapshots(retention_days=30, max_bytes=1024 * 1024, orphan_grace_hours=24)

    refreshed = alert_store.get_alert(alert["id"])
    assert result["detachedAlerts"] == 1
    assert refreshed["snapshotUrl"] is None
    assert refreshed["cleanSnapshotUrl"] is None


def test_cleanup_snapshots_deletes_old_orphans():
    orphan = _test_snapshots / "orphan.jpg"
    orphan.write_bytes(b"orphan")
    old_time = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    os.utime(orphan, (old_time, old_time))

    result = alert_store.cleanup_snapshots(retention_days=30, max_bytes=1024 * 1024, orphan_grace_hours=24)

    assert result["deletedFiles"] == 1
    assert not orphan.exists()
