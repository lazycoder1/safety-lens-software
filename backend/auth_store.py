"""
User authentication and management for Rakshak Lens backend.
"""

import fcntl
import logging
import os
import secrets
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

import bcrypt
import jwt
from psycopg2.extras import RealDictCursor

from db import get_conn
from runtime_storage import STATE_DIR, atomic_write_private

logger = logging.getLogger("rakshak_lens.auth")


_JWT_SECRET_FILE_VALUE = os.environ.get("JWT_SECRET_FILE", "").strip()
JWT_SECRET_PATH = Path(
    _JWT_SECRET_FILE_VALUE or str(STATE_DIR / "auth" / "jwt.secret")
).expanduser()
_JWT_SECRET_MIN_LENGTH = 32
_jwt_secret_lock = threading.Lock()


def _validate_jwt_secret(secret: str, source: str) -> str:
    if secret != secret.strip() or any(char in secret for char in ("\r", "\n", "\x00")):
        raise RuntimeError(f"JWT secret from {source} must not contain surrounding whitespace or line breaks")
    if len(secret.encode("utf-8")) < _JWT_SECRET_MIN_LENGTH:
        raise RuntimeError(
            f"JWT secret from {source} must be at least {_JWT_SECRET_MIN_LENGTH} bytes"
        )
    return secret


def _read_jwt_secret_file(path: Path, source: str) -> str:
    secret = path.read_text()
    if secret.endswith("\r\n"):
        secret = secret[:-2]
    elif secret.endswith("\n"):
        secret = secret[:-1]
    return _validate_jwt_secret(secret, source)


def _load_jwt_secret(seed_secret: str | None = None) -> str:
    """Load an explicit secret or create one once in the persistent state dir."""
    env = os.environ.get("JWT_SECRET", "")
    if env:
        return _validate_jwt_secret(env, "JWT_SECRET")

    explicit_file = os.environ.get("JWT_SECRET_FILE", "").strip()
    if explicit_file:
        if not JWT_SECRET_PATH.is_file():
            raise RuntimeError(f"JWT_SECRET_FILE does not exist: {JWT_SECRET_PATH}")
        return _read_jwt_secret_file(JWT_SECRET_PATH, "JWT_SECRET_FILE")

    JWT_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(JWT_SECRET_PATH.parent, 0o700)
    lock_path = JWT_SECRET_PATH.with_suffix(JWT_SECRET_PATH.suffix + ".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.chmod(lock_path, 0o600)
    try:
        with os.fdopen(lock_fd, "r+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if JWT_SECRET_PATH.is_file():
                os.chmod(JWT_SECRET_PATH, 0o600)
                secret = _read_jwt_secret_file(JWT_SECRET_PATH, str(JWT_SECRET_PATH))
                if seed_secret is not None:
                    validated_seed = _validate_jwt_secret(seed_secret, "legacy config")
                    if not secrets.compare_digest(secret, validated_seed):
                        raise RuntimeError(
                            "Persistent JWT secret does not match legacy application config; "
                            "refusing to invalidate tokens or share deployment identity"
                        )
            else:
                secret = seed_secret or secrets.token_hex(32)
                _validate_jwt_secret(secret, "legacy config")
                atomic_write_private(JWT_SECRET_PATH, (secret + "\n").encode())
                logger.info(
                    "Stored JWT secret in persistent runtime state",
                    extra={"migrated": seed_secret is not None},
                )
            return _validate_jwt_secret(secret, str(JWT_SECRET_PATH))
    finally:
        # fdopen owns lock_fd on the normal path; only close it if fdopen failed.
        try:
            os.close(lock_fd)
        except OSError:
            pass


JWT_SECRET: str | None = None
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 8


def init_jwt_secret() -> str:
    """Initialize the process secret and migrate the legacy config value once."""
    global JWT_SECRET
    if JWT_SECRET is not None:
        return JWT_SECRET
    with _jwt_secret_lock:
        if JWT_SECRET is not None:
            return JWT_SECRET

        from config_manager import get_config, remove_legacy_jwt_secret

        config = get_config()
        auth_config = config.get("auth", {})
        legacy_secret = auth_config.get("jwt_secret") if isinstance(auth_config, dict) else None
        JWT_SECRET = _load_jwt_secret(seed_secret=legacy_secret)

        # JWT material has a dedicated ownership boundary now. Remove only the
        # matching legacy field, never rewrite a stale whole-config snapshot.
        if isinstance(auth_config, dict) and "jwt_secret" in auth_config:
            if remove_legacy_jwt_secret(legacy_secret):
                logger.info("Removed legacy JWT secret from application config")
        return JWT_SECRET


def _get_jwt_secret() -> str:
    return JWT_SECRET if JWT_SECRET is not None else init_jwt_secret()


# ── JWT ─────────────────────────────────────────────────────────────────────

def create_token(user_id: str, username: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])


# ── Password hashing ────────────────────────────────────────────────────────

BCRYPT_MAX_PASSWORD_BYTES = 72
_PASSWORD_TOO_LONG_ERROR = "Password must be at most 72 bytes when encoded as UTF-8"
_PASSWORD_INVALID_ENCODING_ERROR = "Password contains invalid Unicode characters"


def _password_bytes(password: str) -> bytes:
    try:
        return password.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(_PASSWORD_INVALID_ENCODING_ERROR) from exc


def _hash_password(password: str) -> str:
    encoded = _password_bytes(password)
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(_PASSWORD_TOO_LONG_ERROR)
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    try:
        encoded = _password_bytes(password)
    except ValueError:
        return False
    # bcrypt <5.0 silently truncated raw input bytes. Preserve verification for
    # hashes created under that behavior while rejecting all new oversized
    # passwords in validate_password() and _hash_password().
    encoded = encoded[:BCRYPT_MAX_PASSWORD_BYTES]
    try:
        return bcrypt.checkpw(encoded, hashed.encode())
    except (TypeError, ValueError):
        logger.warning("Unable to verify an invalid stored password hash")
        return False


_SPECIAL_CHARS = r"!@#$%^&*()\-_+=\[\]{}|;:,.<>?"


def validate_password(password: str) -> tuple[bool, str | None]:
    """Validate password strength. Returns (is_valid, error_message)."""
    import re
    try:
        encoded = _password_bytes(password)
    except ValueError:
        return False, _PASSWORD_INVALID_ENCODING_ERROR
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        return False, _PASSWORD_TOO_LONG_ERROR
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit"
    if not re.search(rf"[{_SPECIAL_CHARS}]", password):
        return False, "Password must contain at least one special character"
    return True, None


def generate_strong_password(length: int = 16) -> str:
    """Generate a password that meets all strength requirements."""
    import random
    import string
    specials = "!@#$%^&*()-_+=[]{}|;:,.<>?"
    chars = string.ascii_letters + string.digits + specials
    password = [
        random.choice(string.ascii_uppercase),
        random.choice(string.ascii_lowercase),
        random.choice(string.digits),
        random.choice(specials),
    ]
    password += [random.choice(chars) for _ in range(length - 4)]
    random.shuffle(password)
    return "".join(password)


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

    legacy_oversized_password = len(_password_bytes(password)) > BCRYPT_MAX_PASSWORD_BYTES

    # Update last_login
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE users
                   SET last_login = %s,
                       must_change_password = CASE WHEN %s THEN TRUE ELSE must_change_password END
                   WHERE id = %s""",
                (now, legacy_oversized_password, row["id"]),
            )
        conn.commit()

    if legacy_oversized_password:
        row = dict(row)
        row["must_change_password"] = True
        logger.info("Marked legacy oversized password for mandatory rotation")

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
