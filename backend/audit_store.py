"""
Structured audit log for SafetyLens administrative actions.
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from psycopg2.extras import RealDictCursor

from db import get_conn
from secret_redaction import redact_sensitive_data

logger = logging.getLogger("safetylens.audit")


def init_db() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT,
                    actor_username TEXT,
                    actor_role TEXT,
                    action TEXT NOT NULL,
                    target_type TEXT,
                    target_id TEXT,
                    outcome TEXT NOT NULL DEFAULT 'success',
                    request_id TEXT,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log(target_type, target_id)")
        conn.commit()


def log_event(
    action: str,
    *,
    actor_id: str | None = None,
    actor_username: str | None = None,
    actor_role: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    details: dict | None = None,
    outcome: str = "success",
    request_id: str | None = None,
) -> dict:
    entry_id = str(uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = redact_sensitive_data(details or {})

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_log (
                    id, timestamp, actor_id, actor_username, actor_role,
                    action, target_type, target_id, outcome, request_id, details
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    entry_id,
                    timestamp,
                    actor_id,
                    actor_username,
                    actor_role,
                    action,
                    target_type,
                    target_id,
                    outcome,
                    request_id,
                    json.dumps(payload),
                ),
            )
        conn.commit()

    logger.info(
        "Audit event recorded",
        extra={
            "action": action,
            "target_type": target_type,
            "target_id": target_id,
            "request_id": request_id,
            "actor_username": actor_username,
        },
    )

    return {
        "id": entry_id,
        "timestamp": timestamp,
        "action": action,
        "targetType": target_type,
        "targetId": target_id,
        "outcome": outcome,
        "details": payload,
        "actorUsername": actor_username,
        "actorRole": actor_role,
        "requestId": request_id,
    }


def get_recent(limit: int = 200) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def build_actor_context(request) -> dict:
    user = getattr(request.state, "user", {}) or {}
    return {
        "actor_id": user.get("sub"),
        "actor_username": user.get("username"),
        "actor_role": user.get("role"),
        "request_id": getattr(request.state, "request_id", None),
    }


def _row_to_dict(row: dict) -> dict:
    raw_details = row.get("details")
    if isinstance(raw_details, str):
        details = json.loads(raw_details)
    else:
        details = raw_details or {}
    details = redact_sensitive_data(details)
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "action": row["action"],
        "targetType": row.get("target_type"),
        "targetId": row.get("target_id"),
        "outcome": row.get("outcome"),
        "details": details,
        "actorUsername": row.get("actor_username"),
        "actorRole": row.get("actor_role"),
        "requestId": row.get("request_id"),
    }
