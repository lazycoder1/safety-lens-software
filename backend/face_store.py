"""Face enrollment and match-log persistence for Rakshak Lens."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from psycopg2.extras import RealDictCursor

from db import get_conn

logger = logging.getLogger("rakshak_lens.faces")

FACE_PHOTOS_DIR = Path(__file__).parent / "face_photos"
FACE_SNAPSHOTS_DIR = Path(__file__).parent / "face_snapshots"
EMBEDDING_DIM = 512
MATCH_THRESHOLD = 0.42

_vector_available = False
_ACTIVE_FACES_CACHE_TTL_SECONDS = 5.0
_active_faces_cache: tuple[dict, ...] | None = None
_active_faces_cache_expires_at = 0.0
_active_faces_cache_lock = threading.RLock()


def invalidate_active_faces_cache() -> None:
    """Make the next matcher reload enrollment state from PostgreSQL."""
    global _active_faces_cache, _active_faces_cache_expires_at
    with _active_faces_cache_lock:
        _active_faces_cache = None
        _active_faces_cache_expires_at = 0.0


def init_db() -> None:
    """Create face enrollment/log tables and enable pgvector when available."""
    global _vector_available
    FACE_PHOTOS_DIR.mkdir(exist_ok=True)
    FACE_SNAPSHOTS_DIR.mkdir(exist_ok=True)

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                _vector_available = True
            except Exception:
                conn.rollback()
                _vector_available = False
                logger.warning("pgvector extension unavailable; falling back to JSONB embedding search")

        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS enrolled_faces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    face_group TEXT NOT NULL,
                    valid_until TEXT,
                    enrolled_at TEXT NOT NULL,
                    consent_method TEXT NOT NULL,
                    consent_confirmed BOOLEAN NOT NULL DEFAULT TRUE,
                    photo_path TEXT,
                    embedding JSONB NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS face_logs (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    matched_face_id TEXT REFERENCES enrolled_faces(id),
                    confidence DOUBLE PRECISION,
                    snapshot_path TEXT,
                    bbox JSONB NOT NULL DEFAULT '{}'::jsonb,
                    quality_reason TEXT
                )
                """
            )
            if _vector_available:
                cur.execute("ALTER TABLE enrolled_faces ADD COLUMN IF NOT EXISTS embedding_vector vector(512)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_embedding_vector ON enrolled_faces USING ivfflat (embedding_vector vector_cosine_ops)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_group ON enrolled_faces(face_group)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_active ON enrolled_faces(active)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_face_logs_timestamp ON face_logs(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_face_logs_camera ON face_logs(camera_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_face_logs_event_type ON face_logs(event_type)")
        conn.commit()
    invalidate_active_faces_cache()


def save_face_photo(face_id: str, image_bytes: bytes) -> str:
    FACE_PHOTOS_DIR.mkdir(exist_ok=True)
    filename = f"{face_id}.jpg"
    (FACE_PHOTOS_DIR / filename).write_bytes(image_bytes)
    return filename


def save_face_snapshot(log_id: str, image_bytes: bytes | None) -> str | None:
    if not image_bytes:
        return None
    FACE_SNAPSHOTS_DIR.mkdir(exist_ok=True)
    filename = f"{log_id}.jpg"
    (FACE_SNAPSHOTS_DIR / filename).write_bytes(image_bytes)
    return filename


def create_enrollment(
    *,
    name: str,
    group: str,
    valid_until: str | None,
    consent_method: str,
    embedding: list[float],
    photo_bytes: bytes | None = None,
) -> dict:
    face_id = str(uuid4())[:8]
    enrolled_at = datetime.now(timezone.utc).isoformat()
    photo_path = save_face_photo(face_id, photo_bytes) if photo_bytes else None
    vector_literal = _embedding_vector_literal(embedding)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if _vector_available:
                cur.execute(
                    """
                    INSERT INTO enrolled_faces (
                        id, name, face_group, valid_until, enrolled_at,
                        consent_method, consent_confirmed, photo_path, embedding, embedding_vector, active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s, %s, %s::vector, TRUE)
                    RETURNING *
                    """,
                    (face_id, name, group, valid_until, enrolled_at, consent_method, photo_path, json.dumps(embedding), vector_literal),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO enrolled_faces (
                        id, name, face_group, valid_until, enrolled_at,
                        consent_method, consent_confirmed, photo_path, embedding, active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s, %s, TRUE)
                    RETURNING *
                    """,
                    (face_id, name, group, valid_until, enrolled_at, consent_method, photo_path, json.dumps(embedding)),
                )
            row = cur.fetchone()
        conn.commit()
    invalidate_active_faces_cache()
    return _face_row_to_dict(row)


def list_faces(*, include_inactive: bool = False) -> list[dict]:
    clause = "1=1" if include_inactive else "active = TRUE"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM enrolled_faces WHERE {clause} ORDER BY enrolled_at DESC")
            return [_face_row_to_dict(row) for row in cur.fetchall()]


def deactivate_face(face_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("UPDATE enrolled_faces SET active = FALSE WHERE id = %s AND active = TRUE RETURNING *", (face_id,))
            row = cur.fetchone()
        conn.commit()
    if row is not None:
        invalidate_active_faces_cache()
    return _face_row_to_dict(row) if row else None


def find_best_match(embedding: list[float], *, threshold: float = MATCH_THRESHOLD) -> dict | None:
    faces = _active_faces_with_embeddings()
    best: dict | None = None
    best_distance = float("inf")
    for face in faces:
        distance = _cosine_distance(embedding, face["embedding"])
        if distance < best_distance:
            best_distance = distance
            best = face
    if not best or best_distance > threshold:
        return None
    confidence = max(0.0, min(100.0, (1.0 - best_distance) * 100.0))
    return {**_face_row_to_dict(best), "distance": best_distance, "confidence": confidence}


def log_face_event(
    *,
    camera_id: str,
    camera_name: str,
    event_type: str,
    matched_face_id: str | None = None,
    confidence: float | None = None,
    bbox: dict | None = None,
    quality_reason: str | None = None,
    snapshot_jpeg: bytes | None = None,
) -> dict:
    log_id = str(uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_path = save_face_snapshot(log_id, snapshot_jpeg)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO face_logs (
                    id, camera_id, camera_name, timestamp, event_type,
                    matched_face_id, confidence, snapshot_path, bbox, quality_reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    log_id,
                    camera_id,
                    camera_name,
                    timestamp,
                    event_type,
                    matched_face_id,
                    confidence,
                    snapshot_path,
                    json.dumps(bbox or {}),
                    quality_reason,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return _log_row_to_dict(row)


def query_logs(
    *,
    camera_id: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    clauses = ["1=1"]
    params: list = []
    if camera_id:
        clauses.append("l.camera_id = %s")
        params.append(camera_id)
    if event_type:
        clauses.append("l.event_type = %s")
        params.append(event_type)
    params.extend([limit, offset])
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT l.*, f.name AS matched_name, f.face_group AS matched_group
                FROM face_logs l
                LEFT JOIN enrolled_faces f ON f.id = l.matched_face_id
                WHERE {' AND '.join(clauses)}
                ORDER BY l.timestamp DESC
                LIMIT %s OFFSET %s
                """,
                params,
            )
            return [_log_row_to_dict(row) for row in cur.fetchall()]


def _active_faces_with_embeddings() -> list[dict]:
    global _active_faces_cache, _active_faces_cache_expires_at
    now = time.monotonic()
    with _active_faces_cache_lock:
        if (
            _active_faces_cache is not None
            and now < _active_faces_cache_expires_at
        ):
            return list(_active_faces_cache)
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM enrolled_faces WHERE active = TRUE")
                rows = cur.fetchall()
        _active_faces_cache = tuple(dict(row) for row in rows)
        _active_faces_cache_expires_at = now + _ACTIVE_FACES_CACHE_TTL_SECONDS
        return list(_active_faces_cache)


def _embedding_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in embedding[:EMBEDDING_DIM]) + "]"


def _parse_embedding(value) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        return [float(item) for item in json.loads(value)]
    return []


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 1.0
    size = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(size))
    norm_a = math.sqrt(sum(float(a[i]) ** 2 for i in range(size)))
    norm_b = math.sqrt(sum(float(b[i]) ** 2 for i in range(size)))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


def _face_row_to_dict(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "group": row["face_group"],
        "validUntil": row.get("valid_until"),
        "enrolledAt": row["enrolled_at"],
        "consentMethod": row["consent_method"],
        "consentConfirmed": row.get("consent_confirmed", True),
        "photoUrl": f"/api/faces/{row['id']}/photo" if row.get("photo_path") else None,
        "active": row.get("active", True),
    }


def _log_row_to_dict(row: dict) -> dict:
    bbox = row.get("bbox")
    if isinstance(bbox, str):
        bbox = json.loads(bbox)
    return {
        "id": row["id"],
        "cameraId": row["camera_id"],
        "cameraName": row["camera_name"],
        "timestamp": row["timestamp"],
        "eventType": row["event_type"],
        "matchedFaceId": row.get("matched_face_id"),
        "personId": row.get("matched_face_id"),
        "personName": row.get("matched_name"),
        "personGroup": row.get("matched_group"),
        "confidence": row.get("confidence"),
        "snapshotUrl": f"/api/faces/logs/{row['id']}/snapshot" if row.get("snapshot_path") else None,
        "bbox": bbox or {},
        "qualityReason": row.get("quality_reason"),
        "isUnknown": row["event_type"] == "face_unknown",
    }
