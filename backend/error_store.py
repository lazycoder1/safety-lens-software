"""
Persistent error log for Rakshak Lens — stores frontend and backend errors
in PostgreSQL so they can be queried remotely via the admin API.
"""

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from psycopg2.extras import RealDictCursor, Json

from db import get_conn

logger = logging.getLogger("rakshak_lens.errors")


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS error_log (
                    id          TEXT PRIMARY KEY,
                    timestamp   TEXT NOT NULL,
                    source      TEXT NOT NULL DEFAULT 'frontend',
                    message     TEXT NOT NULL,
                    stack       TEXT,
                    url         TEXT,
                    request_id  TEXT,
                    context     JSONB DEFAULT '{}'
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_error_log_ts ON error_log (timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_error_log_source ON error_log (source)")
        conn.commit()
    logger.info("Error log database initialized")


def log_error(
    source: str,
    message: str,
    stack: str | None = None,
    url: str | None = None,
    request_id: str | None = None,
    context: dict | None = None,
) -> dict:
    error_id = uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO error_log (id, timestamp, source, message, stack, url, request_id, context)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (error_id, now, source, message, stack, url, request_id, Json(context or {})),
            )
        conn.commit()
    return {"id": error_id, "timestamp": now, "source": source, "message": message}


def get_errors(
    source: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses = []
    params: list = []
    if source:
        clauses.append("source = %s")
        params.append(source)
    if since:
        clauses.append("timestamp >= %s")
        params.append(since)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM error_log {where} ORDER BY timestamp DESC LIMIT %s",
                params,
            )
            return [dict(r) for r in cur.fetchall()]


def cleanup_old_errors(retention_days: int = 30) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM error_log WHERE timestamp < %s", (cutoff,))
            count = cur.rowcount
        conn.commit()
    if count:
        logger.info("Cleaned up %d error log entries older than %d days", count, retention_days)
    return count
