"""
Alert persistence with PostgreSQL for SafetyLens backend.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from psycopg2.extras import RealDictCursor

from db import get_conn

logger = logging.getLogger("safetylens.alerts")

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def init_db():
    """Create alert tables (pool must already be initialized via db.init_pool)."""
    SNAPSHOTS_DIR.mkdir(exist_ok=True)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    rule TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    zone TEXT NOT NULL DEFAULT 'Unknown',
                    confidence DOUBLE PRECISION NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'YOLO',
                    description TEXT NOT NULL DEFAULT '',
                    snapshot_path TEXT,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    snoozed_until TEXT,
                    false_positive BOOLEAN NOT NULL DEFAULT FALSE
                )
            """)
            # Migration: add columns for violation bboxes and clean snapshots
            cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS bboxes JSONB DEFAULT '[]'")
            cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS clean_snapshot_path TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id)")
        conn.commit()
    logger.info("Database initialized")




def create_alert(
    camera_id: str,
    camera_name: str,
    zone: str,
    rule: str,
    severity: str,
    confidence: float,
    description: str = "",
    source: str = "YOLO",
    snapshot_jpeg: bytes | None = None,
    bboxes: list[dict] | None = None,
    clean_snapshot_jpeg: bytes | None = None,
) -> dict:
    alert_id = str(uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_path = None
    clean_snapshot_path = None
    bboxes = bboxes or []

    if snapshot_jpeg:
        snapshot_filename = f"{alert_id}.jpg"
        (SNAPSHOTS_DIR / snapshot_filename).write_bytes(snapshot_jpeg)
        snapshot_path = snapshot_filename

    if clean_snapshot_jpeg:
        clean_filename = f"{alert_id}_clean.jpg"
        (SNAPSHOTS_DIR / clean_filename).write_bytes(clean_snapshot_jpeg)
        clean_snapshot_path = clean_filename

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO alerts
                   (id, severity, status, rule, camera_id, camera_name, zone, confidence, timestamp, source, description, snapshot_path, bboxes, clean_snapshot_path)
                   VALUES (%s, %s, 'active', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (alert_id, severity, rule, camera_id, camera_name, zone, round(confidence, 2), timestamp, source, description, snapshot_path, json.dumps(bboxes), clean_snapshot_path),
            )
        conn.commit()

    logger.debug("Alert created", extra={"alert_id": alert_id, "camera_id": camera_id})

    return _build_dict(
        alert_id, severity, "active", rule, camera_id, camera_name, zone,
        round(confidence, 2), timestamp, source, description, snapshot_path,
        bboxes=bboxes, clean_snapshot_path=clean_snapshot_path,
    )


def get_alerts(
    severity: str | None = None,
    status: str | None = None,
    camera_id: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict]:
    clauses = ["1=1"]
    params: list = []

    if severity:
        clauses.append("severity = %s")
        params.append(severity)
    if status:
        clauses.append("status = %s")
        params.append(status)
    if camera_id:
        clauses.append("camera_id = %s")
        params.append(camera_id)

    query = f"SELECT * FROM alerts WHERE {' AND '.join(clauses)} ORDER BY timestamp DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [_row_to_dict(row) for row in cur.fetchall()]


def get_alert(alert_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM alerts WHERE id = %s", (alert_id,))
            row = cur.fetchone()
            return _row_to_dict(row) if row else None


def acknowledge_alert(alert_id: str, by: str = "Admin") -> dict | None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET status = 'acknowledged', acknowledged_by = %s, acknowledged_at = %s WHERE id = %s AND status = 'active'",
                (by, now, alert_id),
            )
        conn.commit()
    return get_alert(alert_id)


def resolve_alert(alert_id: str) -> dict | None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET status = 'resolved', resolved_at = %s WHERE id = %s AND status IN ('active', 'acknowledged')",
                (now, alert_id),
            )
        conn.commit()
    return get_alert(alert_id)


def snooze_alert(alert_id: str, minutes: int = 15) -> dict | None:
    until = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET status = 'snoozed', snoozed_until = %s WHERE id = %s AND status IN ('active', 'acknowledged')",
                (until, alert_id),
            )
        conn.commit()
    return get_alert(alert_id)


def mark_false_positive(alert_id: str) -> dict | None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET status = 'resolved', resolved_at = %s, false_positive = TRUE WHERE id = %s",
                (now, alert_id),
            )
        conn.commit()
    return get_alert(alert_id)


def get_stats() -> dict:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM alerts")
            total = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'active'")
            active = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'acknowledged'")
            acknowledged = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'resolved'")
            resolved = cur.fetchone()["cnt"]

            by_severity = {}
            cur.execute("SELECT severity, COUNT(*) as cnt FROM alerts GROUP BY severity")
            for row in cur.fetchall():
                by_severity[row["severity"]] = row["cnt"]

            by_rule = {}
            cur.execute("SELECT rule, COUNT(*) as cnt FROM alerts GROUP BY rule ORDER BY cnt DESC")
            for row in cur.fetchall():
                by_rule[row["rule"]] = row["cnt"]

            by_zone = {}
            cur.execute("SELECT zone, COUNT(*) as cnt FROM alerts GROUP BY zone ORDER BY cnt DESC")
            for row in cur.fetchall():
                by_zone[row["zone"]] = row["cnt"]

            by_camera = {}
            cur.execute("SELECT camera_name, COUNT(*) as cnt FROM alerts GROUP BY camera_name ORDER BY cnt DESC")
            for row in cur.fetchall():
                by_camera[row["camera_name"]] = row["cnt"]

    return {
        "total": total,
        "active": active,
        "acknowledged": acknowledged,
        "resolved": resolved,
        "bySeverity": by_severity,
        "byRule": by_rule,
        "byZone": by_zone,
        "byCamera": by_camera,
    }


def get_time_series(hours: int = 24) -> list[dict]:
    """Return hourly alert counts by severity for the last N hours."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT
                    date_trunc('hour', timestamp::timestamp) as hour,
                    severity,
                    COUNT(*) as count
                FROM alerts
                WHERE timestamp >= %s
                GROUP BY hour, severity
                ORDER BY hour""",
                (since,),
            )
            rows = cur.fetchall()

    # Pivot into [{hour: "...", P1: N, P2: N, P3: N, P4: N}, ...]
    hourly: dict[str, dict] = {}
    for row in rows:
        h = row["hour"].isoformat() if row["hour"] else ""
        if h not in hourly:
            hourly[h] = {"hour": h, "P1": 0, "P2": 0, "P3": 0, "P4": 0}
        hourly[h][row["severity"]] = row["count"]

    return list(hourly.values())


def _row_to_dict(row: dict) -> dict:
    snapshot = row["snapshot_path"]
    raw_bboxes = row.get("bboxes")
    if isinstance(raw_bboxes, str):
        bboxes = json.loads(raw_bboxes)
    elif isinstance(raw_bboxes, list):
        bboxes = raw_bboxes
    else:
        bboxes = []
    return _build_dict(
        row["id"], row["severity"], row["status"], row["rule"],
        row["camera_id"], row["camera_name"], row["zone"],
        row["confidence"], row["timestamp"], row["source"],
        row["description"], snapshot,
        row.get("acknowledged_by"), row.get("acknowledged_at"),
        row.get("resolved_at"), row.get("snoozed_until"), bool(row.get("false_positive", False)),
        bboxes=bboxes, clean_snapshot_path=row.get("clean_snapshot_path"),
    )


def _build_dict(
    id, severity, status, rule, camera_id, camera_name, zone,
    confidence, timestamp, source, description, snapshot_path=None,
    acknowledged_by=None, acknowledged_at=None, resolved_at=None,
    snoozed_until=None, false_positive=False,
    bboxes=None, clean_snapshot_path=None,
) -> dict:
    return {
        "id": id,
        "severity": severity,
        "status": status,
        "rule": rule,
        "cameraId": camera_id,
        "cameraName": camera_name,
        "zone": zone,
        "confidence": confidence,
        "timestamp": timestamp,
        "source": source,
        "description": description,
        "snapshotUrl": f"/api/snapshots/{snapshot_path}" if snapshot_path else None,
        "cleanSnapshotUrl": f"/api/snapshots/{clean_snapshot_path}" if clean_snapshot_path else None,
        "bboxes": bboxes or [],
        "acknowledgedBy": acknowledged_by,
        "acknowledgedAt": acknowledged_at,
        "resolvedAt": resolved_at,
        "snoozedUntil": snoozed_until,
        "falsePositive": false_positive,
    }
