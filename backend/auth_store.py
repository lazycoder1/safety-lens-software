"""
User authentication and management for SafetyLens backend.
"""

import logging
import os
import secrets
from datetime import datetime, timezone, timedelta
from uuid import uuid4

import bcrypt
import jwt
from psycopg2.extras import RealDictCursor

from db import get_conn

logger = logging.getLogger("safetylens.auth")

JWT_SECRET = os.environ.get("JWT_SECRET") or secrets.token_hex(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 8

if not os.environ.get("JWT_SECRET"):
    logger.warning("JWT_SECRET not set — using random key (tokens invalidate on restart)")


# ── JWT ─────────────────────────────────────────────────────────────────────

def create_token(user_id: str, username: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ── Password hashing ────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── Database ────────────────────────────────────────────────────────────────

def init_auth_db():
    """Create users table and seed default admin if none exists."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    status TEXT NOT NULL DEFAULT 'pending',
                    must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            """)
        conn.commit()

    # Seed admin user if no admin exists
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            if cur.fetchone()[0] == 0:
                admin_id = str(uuid4())[:8]
                now = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    """INSERT INTO users (id, username, password_hash, role, status, must_change_password, created_at)
                       VALUES (%s, %s, %s, 'admin', 'active', TRUE, %s)
                       ON CONFLICT (username) DO NOTHING""",
                    (admin_id, "admin", _hash_password("admin123"), now),
                )
                conn.commit()
                logger.info("Default admin user seeded (username: admin)")
            else:
                logger.info("Admin user already exists, skipping seed")

    logger.info("Auth database initialized")


def _user_dict(row: dict) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "status": row["status"],
        "mustChangePassword": bool(row["must_change_password"]),
        "createdAt": row["created_at"],
        "lastLogin": row.get("last_login"),
    }


# ── CRUD ────────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict or None. Only active users can login."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            row = cur.fetchone()

    if not row:
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    if row["status"] != "active":
        return None

    # Update last_login
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET last_login = %s WHERE id = %s", (now, row["id"]))
        conn.commit()

    return _user_dict(row)


def create_user(username: str, password: str, role: str = "viewer", status: str = "pending") -> dict:
    user_id = str(uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO users (id, username, password_hash, role, status, must_change_password, created_at)
                   VALUES (%s, %s, %s, %s, %s, FALSE, %s)""",
                (user_id, username, _hash_password(password), role, status, now),
            )
        conn.commit()

    return {"id": user_id, "username": username, "role": role, "status": status}


def change_password(user_id: str, current_password: str, new_password: str) -> bool:
    """Change password. Returns True on success, False if current password is wrong."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()

    if not row or not _verify_password(current_password, row["password_hash"]):
        return False

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s, must_change_password = FALSE WHERE id = %s",
                (_hash_password(new_password), user_id),
            )
        conn.commit()
    return True


def reset_password(user_id: str, new_password: str) -> bool:
    """Admin resets a user's password. Sets must_change_password=TRUE."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s, must_change_password = TRUE WHERE id = %s",
                (_hash_password(new_password), user_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def get_user(user_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
    return _user_dict(row) if row else None


def get_users() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users ORDER BY created_at DESC")
            return [_user_dict(row) for row in cur.fetchall()]


def update_user_status(user_id: str, status: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET status = %s WHERE id = %s", (status, user_id))
        conn.commit()
    return get_user(user_id)


def update_user_role(user_id: str, role: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET role = %s WHERE id = %s", (role, user_id))
        conn.commit()
    return get_user(user_id)


def delete_user(user_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted
