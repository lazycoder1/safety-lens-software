"""
Shared PostgreSQL connection pool for Rakshak Lens backend.
Thread-safe via psycopg2 ThreadedConnectionPool.
"""

import logging
import os
import threading
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool

logger = logging.getLogger("rakshak_lens.db")

_pool: pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_database_url() -> str:
    """Resolve database URL: env var > config > default."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    try:
        from config_manager import get_config
        cfg = get_config()
        url = cfg.get("database", {}).get("url")
        if url:
            return url
    except Exception:
        pass
    return "postgresql://localhost:5432/rakshak_lens"


def init_pool():
    """Initialize the shared connection pool (idempotent)."""
    global _pool
    db_url = _get_database_url()
    logger.info("Connecting to PostgreSQL", extra={"source": db_url.split("@")[-1] if "@" in db_url else db_url})
    with _pool_lock:
        if _pool is not None:
            return
        _pool = pool.ThreadedConnectionPool(minconn=2, maxconn=10, dsn=db_url)


def close_pool():
    """Graceful shutdown of the connection pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


def check_connection() -> bool:
    """Quick readiness check for health/diagnostics."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        logger.exception("Database health check failed")
        return False


@contextmanager
def get_conn():
    """Check out a connection from the pool, return it when done."""
    global _pool
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)
