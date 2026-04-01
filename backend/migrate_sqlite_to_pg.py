"""
One-time migration: SQLite alerts.db → PostgreSQL.
Usage: DATABASE_URL=postgresql://... python migrate_sqlite_to_pg.py
"""

import os
import sqlite3
import sys
from pathlib import Path

import psycopg2

SQLITE_DB = Path(__file__).parent / "alerts.db"


def migrate():
    if not SQLITE_DB.exists():
        print("No SQLite DB found at", SQLITE_DB)
        return

    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/safetylens")
    print(f"Migrating from {SQLITE_DB} → {db_url}")

    sqlite_conn = sqlite3.connect(str(SQLITE_DB))
    sqlite_conn.row_factory = sqlite3.Row

    pg_conn = psycopg2.connect(db_url)
    pg_cur = pg_conn.cursor()

    rows = sqlite_conn.execute("SELECT * FROM alerts").fetchall()
    print(f"Found {len(rows)} alerts in SQLite")

    migrated = 0
    skipped = 0
    for row in rows:
        try:
            pg_cur.execute(
                """INSERT INTO alerts
                   (id, severity, status, rule, camera_id, camera_name, zone, confidence, timestamp, source, description, snapshot_path, acknowledged_by, acknowledged_at, resolved_at, snoozed_until, false_positive)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO NOTHING""",
                (
                    row["id"], row["severity"], row["status"], row["rule"],
                    row["camera_id"], row["camera_name"], row["zone"],
                    row["confidence"], row["timestamp"], row["source"],
                    row["description"], row["snapshot_path"],
                    row["acknowledged_by"], row["acknowledged_at"],
                    row["resolved_at"], row["snoozed_until"],
                    bool(row["false_positive"]),
                ),
            )
            migrated += 1
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Skip {row['id']}: {e}")

    pg_conn.commit()
    pg_cur.close()
    pg_conn.close()
    sqlite_conn.close()

    print(f"Done: {migrated} migrated, {skipped} skipped")


if __name__ == "__main__":
    migrate()
