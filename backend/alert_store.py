"""
Alert persistence with PostgreSQL for Rakshak Lens backend.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from psycopg2.extras import RealDictCursor

from db import get_conn

logger = logging.getLogger("rakshak_lens.alerts")

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
_get_conn = get_conn


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
            cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS delivery_results JSONB DEFAULT '[]'")
            cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS policy_id TEXT")
            cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS priority INTEGER")
            cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS message TEXT")
            cur.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_bboxes_gin ON alerts USING gin(bboxes)")
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
    policy_id: str | None = None,
    priority: int | None = None,
    message: str | None = None,
    metadata: dict | None = None,
) -> dict:
    alert_id = str(uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_path = None
    clean_snapshot_path = None
    bboxes = bboxes or []

    if snapshot_jpeg:
        try:
            SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_filename = f"{alert_id}.jpg"
            (SNAPSHOTS_DIR / snapshot_filename).write_bytes(snapshot_jpeg)
            snapshot_path = snapshot_filename
        except Exception:
            logger.exception("Failed to write snapshot")

    if clean_snapshot_jpeg:
        try:
            clean_filename = f"{alert_id}_clean.jpg"
            (SNAPSHOTS_DIR / clean_filename).write_bytes(clean_snapshot_jpeg)
            clean_snapshot_path = clean_filename
        except Exception:
            logger.exception("Failed to write clean snapshot")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO alerts
                   (id, severity, status, rule, camera_id, camera_name, zone, confidence, timestamp, source, description, snapshot_path, bboxes, clean_snapshot_path, policy_id, priority, message, metadata)
                   VALUES (%s, %s, 'active', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    alert_id, severity, rule, camera_id, camera_name, zone,
                    round(confidence, 2), timestamp, source, description,
                    snapshot_path, json.dumps(bboxes), clean_snapshot_path,
                    policy_id, priority, message, json.dumps(metadata or {}),
                ),
            )
        conn.commit()

    logger.debug("Alert created", extra={"alert_id": alert_id, "camera_id": camera_id})

    return _build_dict(
        alert_id, severity, "active", rule, camera_id, camera_name, zone,
        round(confidence, 2), timestamp, source, description, snapshot_path,
        bboxes=bboxes, clean_snapshot_path=clean_snapshot_path,
        policy_id=policy_id, priority=priority, message=message, metadata=metadata or {},
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


def update_delivery_results(alert_id: str, results: list[dict]) -> dict | None:
    """Persist per-output alert delivery results."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET delivery_results = %s WHERE id = %s",
                (json.dumps(results), alert_id),
            )
        conn.commit()
    return get_alert(alert_id)


def get_stats(camera_id: str | None = None) -> dict:
    cam_filter = ""
    cam_active_filter = " WHERE status = 'active'"
    params: list = []
    cam_params: list = []
    if camera_id:
        cam_filter = " WHERE camera_id = %s"
        cam_active_filter = " WHERE status = 'active' AND camera_id = %s"
        params = [camera_id]
        cam_params = [camera_id]

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM alerts" + cam_filter, params)
            total = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM alerts" + cam_active_filter, cam_params)
            active = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'acknowledged'" +
                        (" AND camera_id = %s" if camera_id else ""), cam_params)
            acknowledged = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM alerts WHERE status = 'resolved'" +
                        (" AND camera_id = %s" if camera_id else ""), cam_params)
            resolved = cur.fetchone()["cnt"]

            by_severity = {}
            cur.execute("SELECT severity, COUNT(*) as cnt FROM alerts" + cam_active_filter +
                        " GROUP BY severity", cam_params)
            for row in cur.fetchall():
                by_severity[row["severity"]] = row["cnt"]

            by_rule = {}
            cur.execute("SELECT rule, COUNT(*) as cnt FROM alerts" + cam_active_filter +
                        " GROUP BY rule ORDER BY cnt DESC", cam_params)
            for row in cur.fetchall():
                by_rule[row["rule"]] = row["cnt"]

            by_zone = {}
            cur.execute("SELECT zone, COUNT(*) as cnt FROM alerts" + cam_active_filter +
                        " GROUP BY zone ORDER BY cnt DESC", cam_params)
            for row in cur.fetchall():
                by_zone[row["zone"]] = row["cnt"]

            by_camera = {}
            cur.execute("SELECT camera_name, COUNT(*) as cnt FROM alerts" + cam_active_filter +
                        " GROUP BY camera_name ORDER BY cnt DESC", cam_params)
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


def get_time_series(hours: int = 24, camera_id: str | None = None) -> list[dict]:
    """Return hourly alert counts by severity for the last N hours."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    cam_clause = " AND camera_id = %s" if camera_id else ""
    params: list = [since] + ([camera_id] if camera_id else [])
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""SELECT
                    date_trunc('hour', timestamp::timestamp) as hour,
                    severity,
                    COUNT(*) as count
                FROM alerts
                WHERE timestamp >= %s{cam_clause}
                GROUP BY hour, severity
                ORDER BY hour""",
                params,
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


def get_compliance_metrics(window_hours: int = 24, camera_id: str | None = None) -> dict:
    """Aggregate safety KPIs over the last N hours.

    - safety_compliance_pct: % of hour-buckets with zero P1/P2 alerts
    - ppe_compliance_pct:    % of hour-buckets with zero Missing-X alerts
    - mtta_seconds:          mean (acknowledged_at - timestamp) for alerts acked in window, or None
    - active_p1_count / active_p2_count: snapshot of current active severities
    """
    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    since_iso = since.isoformat()
    cam_clause = " AND camera_id = %s" if camera_id else ""
    cam_params: list = [camera_id] if camera_id else []

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""SELECT
                    date_trunc('hour', timestamp::timestamp) AS hour,
                    bool_or(severity IN ('P1','P2')) AS has_critical,
                    bool_or(rule LIKE %s) AS has_ppe_miss
                FROM alerts
                WHERE timestamp >= %s{cam_clause}
                GROUP BY hour""",
                ["Missing %", since_iso] + cam_params,
            )
            bucket_rows = cur.fetchall()
            bad_safety_hours = sum(1 for r in bucket_rows if r["has_critical"])
            bad_ppe_hours = sum(1 for r in bucket_rows if r["has_ppe_miss"])

            cur.execute(
                f"""SELECT AVG(EXTRACT(EPOCH FROM (acknowledged_at::timestamp - timestamp::timestamp))) AS mtta
                FROM alerts
                WHERE acknowledged_at IS NOT NULL AND timestamp >= %s{cam_clause}""",
                [since_iso] + cam_params,
            )
            mtta_row = cur.fetchone()
            mtta = mtta_row["mtta"] if mtta_row and mtta_row["mtta"] is not None else None

            cur.execute(
                "SELECT severity, COUNT(*) AS cnt FROM alerts WHERE status='active'" +
                (" AND camera_id = %s" if camera_id else "") +
                " GROUP BY severity",
                cam_params,
            )
            active_by_sev = {row["severity"]: row["cnt"] for row in cur.fetchall()}

    total_hours = max(window_hours, 1)
    safety_pct = round(100.0 * (total_hours - bad_safety_hours) / total_hours, 1)
    ppe_pct = round(100.0 * (total_hours - bad_ppe_hours) / total_hours, 1)

    # Previous period for trend comparison
    prev_start = since - timedelta(hours=window_hours)
    prev_start_iso = prev_start.isoformat()
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""SELECT
                    date_trunc('hour', timestamp::timestamp) AS hour,
                    bool_or(severity IN ('P1','P2')) AS has_critical,
                    bool_or(rule LIKE %s) AS has_ppe_miss
                FROM alerts
                WHERE timestamp >= %s AND timestamp < %s{cam_clause}
                GROUP BY hour""",
                ["Missing %", prev_start_iso, since_iso] + cam_params,
            )
            prev_rows = cur.fetchall()
            prev_bad_safety = sum(1 for r in prev_rows if r["has_critical"])
            prev_bad_ppe = sum(1 for r in prev_rows if r["has_ppe_miss"])

            cur.execute(
                f"""SELECT AVG(EXTRACT(EPOCH FROM (acknowledged_at::timestamp - timestamp::timestamp))) AS mtta
                FROM alerts
                WHERE acknowledged_at IS NOT NULL AND timestamp >= %s AND timestamp < %s{cam_clause}""",
                [prev_start_iso, since_iso] + cam_params,
            )
            prev_mtta_row = cur.fetchone()
            prev_mtta = prev_mtta_row["mtta"] if prev_mtta_row and prev_mtta_row["mtta"] is not None else None

    prev_safety_pct = round(100.0 * (total_hours - prev_bad_safety) / total_hours, 1)
    prev_ppe_pct = round(100.0 * (total_hours - prev_bad_ppe) / total_hours, 1)

    return {
        "safety_compliance_pct": max(0.0, min(100.0, safety_pct)),
        "ppe_compliance_pct": max(0.0, min(100.0, ppe_pct)),
        "mtta_seconds": round(float(mtta), 1) if mtta is not None else None,
        "active_p1_count": int(active_by_sev.get("P1", 0)),
        "active_p2_count": int(active_by_sev.get("P2", 0)),
        "window_hours": window_hours,
        "prev_safety_compliance_pct": max(0.0, min(100.0, prev_safety_pct)),
        "prev_ppe_compliance_pct": max(0.0, min(100.0, prev_ppe_pct)),
        "prev_mtta_seconds": round(float(prev_mtta), 1) if prev_mtta is not None else None,
    }


def get_snapshot_usage() -> dict:
    total_bytes = 0
    total_files = 0
    if SNAPSHOTS_DIR.exists():
        for path in SNAPSHOTS_DIR.iterdir():
            if path.is_file():
                total_files += 1
                total_bytes += path.stat().st_size
    return {
        "dir": str(SNAPSHOTS_DIR),
        "files": total_files,
        "bytes": total_bytes,
    }


def cleanup_snapshots(
    *,
    retention_days: int,
    max_bytes: int,
    orphan_grace_hours: int,
) -> dict:
    """Prune stale or oversized snapshot files and detach them from alert rows.

    Policy:
    - Active/acknowledged/snoozed alerts keep snapshots.
    - Resolved alerts lose snapshots after `retention_days`.
    - If the directory still exceeds `max_bytes`, prune oldest resolved snapshots first.
    - Unreferenced orphan files older than `orphan_grace_hours` are deleted.
    """
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    retention_cutoff = now - timedelta(days=max(retention_days, 0))
    orphan_cutoff = now - timedelta(hours=max(orphan_grace_hours, 0))
    deleted_files = 0
    reclaimed_bytes = 0
    detached_alerts = 0

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, status, timestamp, snapshot_path, clean_snapshot_path
                FROM alerts
                WHERE snapshot_path IS NOT NULL OR clean_snapshot_path IS NOT NULL
                ORDER BY timestamp ASC
                """
            )
            rows = cur.fetchall()

            # Age-based pruning for resolved alerts.
            for row in rows:
                ts = _parse_iso_timestamp(row["timestamp"])
                if row["status"] != "resolved" or ts >= retention_cutoff:
                    continue
                deleted, bytes_freed = _delete_snapshot_pair(row.get("snapshot_path"), row.get("clean_snapshot_path"))
                if deleted:
                    cur.execute(
                        "UPDATE alerts SET snapshot_path = NULL, clean_snapshot_path = NULL WHERE id = %s",
                        (row["id"],),
                    )
                    detached_alerts += 1
                    deleted_files += deleted
                    reclaimed_bytes += bytes_freed

            conn.commit()

            # Capacity-based pruning for remaining resolved alerts.
            usage = get_snapshot_usage()
            if usage["bytes"] > max_bytes:
                cur.execute(
                    """
                    SELECT id, timestamp, snapshot_path, clean_snapshot_path
                    FROM alerts
                    WHERE status = 'resolved'
                      AND (snapshot_path IS NOT NULL OR clean_snapshot_path IS NOT NULL)
                    ORDER BY timestamp ASC
                    """
                )
                for row in cur.fetchall():
                    if usage["bytes"] <= max_bytes:
                        break
                    deleted, bytes_freed = _delete_snapshot_pair(row.get("snapshot_path"), row.get("clean_snapshot_path"))
                    if not deleted:
                        continue
                    cur.execute(
                        "UPDATE alerts SET snapshot_path = NULL, clean_snapshot_path = NULL WHERE id = %s",
                        (row["id"],),
                    )
                    detached_alerts += 1
                    deleted_files += deleted
                    reclaimed_bytes += bytes_freed
                    usage["bytes"] = max(0, usage["bytes"] - bytes_freed)
                conn.commit()

            # Orphan cleanup, with a grace window to avoid racing fresh writes.
            cur.execute(
                """
                SELECT snapshot_path, clean_snapshot_path
                FROM alerts
                WHERE snapshot_path IS NOT NULL OR clean_snapshot_path IS NOT NULL
                """
            )
            referenced = set()
            for row in cur.fetchall():
                for name in (row.get("snapshot_path"), row.get("clean_snapshot_path")):
                    if name:
                        referenced.add(name)

    for path in SNAPSHOTS_DIR.iterdir():
        if not path.is_file() or path.name in referenced:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified >= orphan_cutoff:
            continue
        reclaimed_bytes += path.stat().st_size
        deleted_files += 1
        path.unlink(missing_ok=True)

    final_usage = get_snapshot_usage()
    return {
        "snapshotRetentionDays": retention_days,
        "snapshotMaxBytes": max_bytes,
        "deletedFiles": deleted_files,
        "reclaimedBytes": reclaimed_bytes,
        "detachedAlerts": detached_alerts,
        "remainingBytes": final_usage["bytes"],
        "remainingFiles": final_usage["files"],
    }


def _row_to_dict(row: dict) -> dict:
    snapshot = row["snapshot_path"]
    raw_bboxes = row.get("bboxes")
    if isinstance(raw_bboxes, str):
        bboxes = json.loads(raw_bboxes)
    elif isinstance(raw_bboxes, list):
        bboxes = raw_bboxes
    else:
        bboxes = []
    raw_delivery_results = row.get("delivery_results")
    if isinstance(raw_delivery_results, str):
        delivery_results = json.loads(raw_delivery_results)
    elif isinstance(raw_delivery_results, list):
        delivery_results = raw_delivery_results
    else:
        delivery_results = []
    raw_metadata = row.get("metadata")
    if isinstance(raw_metadata, str):
        metadata = json.loads(raw_metadata)
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        metadata = {}
    return _build_dict(
        row["id"], row["severity"], row["status"], row["rule"],
        row["camera_id"], row["camera_name"], row["zone"],
        row["confidence"], row["timestamp"], row["source"],
        row["description"], snapshot,
        row.get("acknowledged_by"), row.get("acknowledged_at"),
        row.get("resolved_at"), row.get("snoozed_until"), bool(row.get("false_positive", False)),
        bboxes=bboxes, clean_snapshot_path=row.get("clean_snapshot_path"),
        delivery_results=delivery_results,
        policy_id=row.get("policy_id"), priority=row.get("priority"),
        message=row.get("message"), metadata=metadata,
    )


def _build_dict(
    id, severity, status, rule, camera_id, camera_name, zone,
    confidence, timestamp, source, description, snapshot_path=None,
    acknowledged_by=None, acknowledged_at=None, resolved_at=None,
    snoozed_until=None, false_positive=False,
    bboxes=None, clean_snapshot_path=None, delivery_results=None,
    policy_id=None, priority=None, message=None, metadata=None,
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
        "deliveryResults": delivery_results or [],
        "policyId": policy_id,
        "priority": priority,
        "message": message,
        "metadata": metadata or {},
        "acknowledgedBy": acknowledged_by,
        "acknowledgedAt": acknowledged_at,
        "resolvedAt": resolved_at,
        "snoozedUntil": snoozed_until,
        "falsePositive": false_positive,
    }


def _delete_snapshot_pair(snapshot_path: str | None, clean_snapshot_path: str | None) -> tuple[int, int]:
    deleted = 0
    reclaimed = 0
    for name in (snapshot_path, clean_snapshot_path):
        if not name:
            continue
        path = SNAPSHOTS_DIR / name
        if not path.exists():
            continue
        reclaimed += path.stat().st_size
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted, reclaimed


def _parse_iso_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── Search & Heatmap functions ────────────────────────────────────────────

_TIME_RANGE_MAP = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}


def get_detection_classes() -> list[str]:
    """Return distinct detected object class names from bboxes JSONB."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT elem->>'class' AS class_name
                FROM alerts, jsonb_array_elements(bboxes) elem
                WHERE elem->>'class' IS NOT NULL
                ORDER BY class_name
            """)
            return [row[0] for row in cur.fetchall()]


def search_alerts(
    query: str | None = None,
    camera_id: str | None = None,
    severity: str | None = None,
    detection_class: str | None = None,
    time_range: str | None = None,
    sort: str = "relevance",
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Search alerts with text matching and filters. Returns {results, total}."""
    clauses: list[str] = ["1=1"]
    params: list = []
    has_text = bool(query and query.strip())

    if has_text:
        q = query.strip()
        clauses.append("""(
            description ILIKE %s
            OR rule ILIKE %s
            OR camera_name ILIKE %s
            OR zone ILIKE %s
            OR EXISTS (
                SELECT 1 FROM jsonb_array_elements(bboxes) elem
                WHERE elem->>'class' ILIKE %s
            )
        )""")
        like = f"%{q}%"
        params.extend([like, like, like, like, like])

    if camera_id:
        clauses.append("camera_id = %s")
        params.append(camera_id)
    if severity:
        clauses.append("severity = %s")
        params.append(severity)
    if detection_class:
        clauses.append("""EXISTS (
            SELECT 1 FROM jsonb_array_elements(bboxes) elem
            WHERE elem->>'class' ILIKE %s
        )""")
        params.append(f"%{detection_class}%")
    if time_range and time_range in _TIME_RANGE_MAP:
        since = (datetime.now(timezone.utc) - timedelta(hours=_TIME_RANGE_MAP[time_range])).isoformat()
        clauses.append("timestamp >= %s")
        params.append(since)

    where = " AND ".join(clauses)

    # Relevance scoring for text queries
    if has_text:
        q = query.strip()
        like = f"%{q}%"
        order_clause = """(
            CASE WHEN description ILIKE %s THEN 3 ELSE 0 END
            + CASE WHEN rule ILIKE %s THEN 2 ELSE 0 END
            + CASE WHEN camera_name ILIKE %s THEN 2 ELSE 0 END
            + CASE WHEN zone ILIKE %s THEN 1 ELSE 0 END
            + CASE WHEN EXISTS (
                SELECT 1 FROM jsonb_array_elements(bboxes) elem
                WHERE elem->>'class' ILIKE %s
            ) THEN 2 ELSE 0 END
        ) DESC, timestamp DESC"""
        order_params = [like, like, like, like, like]
    else:
        order_clause = "timestamp DESC"
        order_params = []

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Count
            cur.execute(f"SELECT COUNT(*) AS cnt FROM alerts WHERE {where}", params)
            total = cur.fetchone()["cnt"]

            # Results
            if sort == "recent" or not has_text:
                final_order = "timestamp DESC"
                final_order_params = []
            else:
                final_order = order_clause
                final_order_params = order_params

            cur.execute(
                f"SELECT * FROM alerts WHERE {where} ORDER BY {final_order} LIMIT %s OFFSET %s",
                params + final_order_params + [limit, offset],
            )
            rows = cur.fetchall()

    results = []
    for row in rows:
        d = _row_to_dict(row)
        # Extract detected classes from bboxes
        classes = list({b.get("class", "") for b in d.get("bboxes", []) if b.get("class")})
        d["detectedClasses"] = sorted(classes)
        results.append(d)

    return {"results": results, "total": total, "limit": limit, "offset": offset}


def find_similar_alerts(alert_id: str, limit: int = 20) -> list[dict]:
    """Find alerts similar to the given alert by rule+camera or shared detected classes."""
    source = get_alert(alert_id)
    if not source:
        return []

    rule = source["rule"]
    camera_id = source["cameraId"]
    classes = [b.get("class", "") for b in source.get("bboxes", []) if b.get("class")]

    clauses = ["id != %s"]
    params: list = [alert_id]

    if classes:
        clauses.append("""(
            (rule = %s AND camera_id = %s)
            OR EXISTS (
                SELECT 1 FROM jsonb_array_elements(bboxes) elem
                WHERE elem->>'class' = ANY(%s)
            )
        )""")
        params.extend([rule, camera_id, classes])
    else:
        clauses.append("(rule = %s AND camera_id = %s)")
        params.extend([rule, camera_id])

    where = " AND ".join(clauses)
    order = """CASE WHEN rule = %s AND camera_id = %s THEN 0 ELSE 1 END, timestamp DESC"""
    order_params = [rule, camera_id]

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM alerts WHERE {where} ORDER BY {order} LIMIT %s",
                params + order_params + [limit],
            )
            rows = cur.fetchall()

    results = []
    for row in rows:
        d = _row_to_dict(row)
        classes_list = list({b.get("class", "") for b in d.get("bboxes", []) if b.get("class")})
        d["detectedClasses"] = sorted(classes_list)
        results.append(d)
    return results


def get_zone_time_heatmap(
    hours: int = 24,
    camera_id: str | None = None,
    severity: str | None = None,
    bucket: str = "hour",
) -> dict:
    """Return zone × time-bucket aggregation for heatmap visualization."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    clauses = ["timestamp >= %s"]
    params: list = [since]

    if camera_id:
        clauses.append("camera_id = %s")
        params.append(camera_id)
    if severity:
        clauses.append("severity = %s")
        params.append(severity)

    where = " AND ".join(clauses)
    trunc = "hour" if bucket == "hour" else "day"

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""SELECT zone, date_trunc(%s, timestamp::timestamp) AS bucket, COUNT(*) AS count
                    FROM alerts WHERE {where}
                    GROUP BY zone, bucket ORDER BY zone, bucket""",
                [trunc] + params,
            )
            rows = cur.fetchall()

    # Collect unique zones and buckets
    zone_set: set[str] = set()
    bucket_set: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        z = row["zone"]
        b = row["bucket"].isoformat() if row["bucket"] else ""
        zone_set.add(z)
        bucket_set.add(b)
        counts[(z, b)] = row["count"]

    zones = sorted(zone_set)
    buckets = sorted(bucket_set)

    # Fill all zone×bucket combos including zeros
    cells = []
    max_count = 0
    for z in zones:
        for b in buckets:
            c = counts.get((z, b), 0)
            cells.append({"zone": z, "bucket": b, "count": c})
            if c > max_count:
                max_count = c

    return {"zones": zones, "buckets": buckets, "cells": cells, "maxCount": max_count}


def get_spatial_heatmap(
    camera_id: str,
    hours: int = 24,
    severity: str | None = None,
    grid_size: int = 10,
) -> dict:
    """Return detection density grid for a camera using bbox centers."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    clauses = ["camera_id = %s", "timestamp >= %s", "bboxes != '[]'::jsonb"]
    params: list = [camera_id, since]

    if severity:
        clauses.append("severity = %s")
        params.append(severity)

    where = " AND ".join(clauses)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT bboxes FROM alerts WHERE {where}", params)
            rows = cur.fetchall()

    # Build grid
    grid = [[0] * grid_size for _ in range(grid_size)]
    total_detections = 0

    for row in rows:
        raw = row["bboxes"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, list):
            continue
        for det in raw:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            # Clamp to 0-1 range (bboxes are normalized)
            cx = max(0.0, min(cx, 0.999))
            cy = max(0.0, min(cy, 0.999))
            col = int(cx * grid_size)
            row_idx = int(cy * grid_size)
            grid[row_idx][col] += 1
            total_detections += 1

    max_count = max((max(r) for r in grid), default=0)
    return {
        "gridSize": grid_size,
        "cells": grid,
        "maxCount": max_count,
        "totalDetections": total_detections,
    }


# ── Auto-resolve stale alerts ─────────────────────────────────────────────

AUTO_RESOLVE_MAX_AGE_HOURS = 24
AUTO_RESOLVE_INTERVAL_HOURS = 1


def auto_resolve_stale_alerts(max_age_hours: int = AUTO_RESOLVE_MAX_AGE_HOURS) -> int:
    """Resolve active alerts older than max_age_hours. Returns count resolved."""
    now = datetime.now(timezone.utc).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET status = 'resolved', resolved_at = %s "
                "WHERE status = 'active' AND timestamp < %s",
                (now, cutoff),
            )
            count = cur.rowcount
        conn.commit()
    if count:
        logger.info("Auto-resolved %d stale alerts older than %dh", count, max_age_hours)
    return count


async def auto_resolve_loop() -> None:
    """Background loop that auto-resolves stale alerts every hour."""
    await asyncio.to_thread(auto_resolve_stale_alerts)  # Run once at startup
    while True:
        await asyncio.sleep(AUTO_RESOLVE_INTERVAL_HOURS * 60 * 60)
        try:
            await asyncio.to_thread(auto_resolve_stale_alerts)
        except Exception:
            logger.exception("Auto-resolve loop failed")
