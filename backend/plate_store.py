"""ANPR plate-list and read-log persistence for Rakshak Lens."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher
from uuid import uuid4

from psycopg2 import errors
from psycopg2.extras import RealDictCursor

from db import get_conn

PLATE_SNAPSHOTS_DIR = Path(__file__).parent / "plate_snapshots"
PLATE_CROPS_DIR = Path(__file__).parent / "plate_crops"

LIST_TYPES = {"whitelist", "blocked", "visitors"}
EVENT_TYPES = {"plate_read", "plate_unknown", "plate_blocked", "plate_visitor", "plate_low_confidence"}
INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
OCR_EQUIVALENT_GROUPS = (
    set("0ODQ"),
    set("1IL"),
    set("2Z"),
    set("5S"),
    set("6G"),
    set("8B"),
    set("HMN"),
)
SIMILAR_PLATE_MIN_SCORE = 0.88


def init_db() -> None:
    PLATE_SNAPSHOTS_DIR.mkdir(exist_ok=True)
    PLATE_CROPS_DIR.mkdir(exist_ok=True)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_lists (
                    id TEXT PRIMARY KEY,
                    plate_text TEXT NOT NULL,
                    normalized_plate TEXT NOT NULL,
                    list_type TEXT NOT NULL,
                    owner_name TEXT NOT NULL DEFAULT '',
                    vehicle_desc TEXT NOT NULL DEFAULT '',
                    valid_from TEXT,
                    valid_until TEXT,
                    created_at TEXT NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS plate_reads (
                    id TEXT PRIMARY KEY,
                    plate_text TEXT NOT NULL,
                    normalized_plate TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    matched_list_id TEXT REFERENCES plate_lists(id),
                    matched_list TEXT,
                    confidence DOUBLE PRECISION,
                    detection_confidence DOUBLE PRECISION,
                    ocr_confidence DOUBLE PRECISION,
                    snapshot_path TEXT,
                    crop_path TEXT,
                    bbox JSONB NOT NULL DEFAULT '{}'::jsonb,
                    vehicle_class TEXT,
                    quality_reason TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_lists_normalized ON plate_lists(normalized_plate)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_lists_type ON plate_lists(list_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_lists_active ON plate_lists(active)")
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_plate_lists_active_unique
                ON plate_lists(normalized_plate, list_type)
                WHERE active = TRUE
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_timestamp ON plate_reads(timestamp DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_camera ON plate_reads(camera_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_normalized ON plate_reads(normalized_plate)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_reads_event_type ON plate_reads(event_type)")
        conn.commit()


def normalize_plate_text(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def is_indian_plate_format(value: str | None) -> bool:
    return bool(INDIAN_PLATE_RE.match(normalize_plate_text(value)))


def list_plate_entries(*, list_type: str | None = None, include_inactive: bool = False) -> list[dict]:
    clauses = ["1=1" if include_inactive else "active = TRUE"]
    params: list = []
    if list_type:
        clauses.append("list_type = %s")
        params.append(list_type)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM plate_lists WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
                params,
            )
            return [_plate_row_to_dict(row) for row in cur.fetchall()]


def create_plate_entry(
    *,
    plate_text: str,
    list_type: str,
    owner_name: str = "",
    vehicle_desc: str = "",
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> dict:
    normalized = _validate_plate_entry(plate_text, list_type)
    now = datetime.now(timezone.utc).isoformat()
    entry_id = str(uuid4())[:8]
    conn = None
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO plate_lists (
                        id, plate_text, normalized_plate, list_type, owner_name,
                        vehicle_desc, valid_from, valid_until, created_at, active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                    RETURNING *
                    """,
                    (
                        entry_id,
                        plate_text.strip().upper(),
                        normalized,
                        list_type,
                        owner_name.strip(),
                        vehicle_desc.strip(),
                        valid_from or None,
                        valid_until or None,
                        now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
    except errors.UniqueViolation as exc:
        if conn:
            conn.rollback()
        raise ValueError("Plate already exists in this active list") from exc
    return _plate_row_to_dict(row)


def update_plate_entry(
    entry_id: str,
    *,
    plate_text: str,
    list_type: str,
    owner_name: str = "",
    vehicle_desc: str = "",
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> dict | None:
    normalized = _validate_plate_entry(plate_text, list_type)
    conn = None
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE plate_lists
                    SET plate_text = %s, normalized_plate = %s, list_type = %s,
                        owner_name = %s, vehicle_desc = %s, valid_from = %s, valid_until = %s
                    WHERE id = %s AND active = TRUE
                    RETURNING *
                    """,
                    (
                        plate_text.strip().upper(),
                        normalized,
                        list_type,
                        owner_name.strip(),
                        vehicle_desc.strip(),
                        valid_from or None,
                        valid_until or None,
                        entry_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
    except errors.UniqueViolation as exc:
        if conn:
            conn.rollback()
        raise ValueError("Plate already exists in this active list") from exc
    return _plate_row_to_dict(row) if row else None


def deactivate_plate_entry(entry_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("UPDATE plate_lists SET active = FALSE WHERE id = %s AND active = TRUE RETURNING *", (entry_id,))
            row = cur.fetchone()
        conn.commit()
    return _plate_row_to_dict(row) if row else None


def import_plate_csv(blob: bytes) -> dict:
    text = blob.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    created: list[dict] = []
    failed: list[dict] = []
    for row_index, row in enumerate(reader, start=2):
        try:
            plate = row.get("plate") or row.get("plateNumber") or row.get("plate_text") or ""
            list_type = _normalize_list_type(row.get("list") or row.get("listType") or row.get("list_type") or "whitelist")
            created.append(
                create_plate_entry(
                    plate_text=plate,
                    list_type=list_type,
                    owner_name=row.get("owner") or row.get("ownerName") or row.get("owner_name") or "",
                    vehicle_desc=row.get("vehicle") or row.get("vehicleDesc") or row.get("vehicle_desc") or "",
                    valid_from=row.get("validFrom") or row.get("valid_from") or None,
                    valid_until=row.get("validUntil") or row.get("valid_until") or None,
                )
            )
        except Exception as exc:
            failed.append({"row": row_index, "error": str(exc), "plate": row.get("plate") or row.get("plateNumber")})
    return {"created": created, "failed": failed}


def find_matching_plate(normalized_plate: str) -> dict | None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM plate_lists
                WHERE active = TRUE
                  AND normalized_plate = %s
                  AND (valid_from IS NULL OR valid_from <= %s)
                  AND (valid_until IS NULL OR valid_until >= %s)
                ORDER BY
                  CASE list_type WHEN 'blocked' THEN 1 WHEN 'whitelist' THEN 2 ELSE 3 END,
                  created_at DESC
                LIMIT 1
                """,
                (normalized_plate, now, now),
            )
            row = cur.fetchone()
    return _plate_row_to_dict(row) if row else None


def find_similar_plate(normalized_plate: str, *, min_score: float = SIMILAR_PLATE_MIN_SCORE) -> dict | None:
    normalized = normalize_plate_text(normalized_plate)
    if len(normalized) < 6:
        return None
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM plate_lists
                WHERE active = TRUE
                  AND (valid_from IS NULL OR valid_from <= %s)
                  AND (valid_until IS NULL OR valid_until >= %s)
                """,
                (now, now),
            )
            rows = cur.fetchall()

    best: tuple[float, dict] | None = None
    for row in rows:
        candidate = _plate_row_to_dict(row)
        score = plate_similarity_score(normalized, candidate["normalizedPlate"])
        if score < min_score:
            continue
        priority_bonus = {"blocked": 0.003, "whitelist": 0.002, "visitors": 0.001}.get(candidate["list"], 0)
        ranked_score = score + priority_bonus
        if best is None or ranked_score > best[0]:
            best = (ranked_score, {**candidate, "similarityScore": round(score, 3)})
    return best[1] if best else None


def plate_similarity_score(left: str, right: str) -> float:
    a = normalize_plate_text(left)
    b = normalize_plate_text(right)
    if not a or not b:
        return 0.0
    base = SequenceMatcher(None, a, b).ratio()
    if len(a) != len(b):
        return base
    mismatches = [(ca, cb) for ca, cb in zip(a, b) if ca != cb]
    if not mismatches:
        return 1.0
    equivalent = sum(1 for ca, cb in mismatches if _is_ocr_equivalent(ca, cb))
    weighted_distance = (len(mismatches) - equivalent) + equivalent * 0.45
    weighted_score = max(0.0, 1.0 - weighted_distance / max(len(a), len(b), 1))
    return max(base, weighted_score)


def log_plate_read(
    *,
    plate_text: str,
    camera_id: str,
    camera_name: str,
    event_type: str,
    matched_list_id: str | None = None,
    matched_list: str | None = None,
    confidence: float | None = None,
    detection_confidence: float | None = None,
    ocr_confidence: float | None = None,
    bbox: dict | None = None,
    vehicle_class: str | None = None,
    quality_reason: str | None = None,
    snapshot_jpeg: bytes | None = None,
    crop_jpeg: bytes | None = None,
) -> dict:
    if event_type not in EVENT_TYPES:
        raise ValueError("Unsupported plate event type")
    read_id = str(uuid4())[:8]
    normalized = normalize_plate_text(plate_text)
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_path = _save_runtime_file(PLATE_SNAPSHOTS_DIR, read_id, snapshot_jpeg)
    crop_path = _save_runtime_file(PLATE_CROPS_DIR, read_id, crop_jpeg)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO plate_reads (
                    id, plate_text, normalized_plate, camera_id, camera_name, timestamp,
                    event_type, matched_list_id, matched_list, confidence,
                    detection_confidence, ocr_confidence, snapshot_path, crop_path,
                    bbox, vehicle_class, quality_reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    read_id,
                    plate_text,
                    normalized,
                    camera_id,
                    camera_name,
                    timestamp,
                    event_type,
                    matched_list_id,
                    matched_list,
                    confidence,
                    detection_confidence,
                    ocr_confidence,
                    snapshot_path,
                    crop_path,
                    json.dumps(bbox or {}),
                    vehicle_class,
                    quality_reason,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return _read_row_to_dict(row)


def query_reads(
    *,
    plate: str | None = None,
    camera_id: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    clauses = ["1=1"]
    params: list = []
    if plate:
        clauses.append("normalized_plate LIKE %s")
        params.append(f"%{normalize_plate_text(plate)}%")
    if camera_id:
        clauses.append("camera_id = %s")
        params.append(camera_id)
    if event_type:
        clauses.append("event_type = %s")
        params.append(event_type)
    params.extend([limit, offset])
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT *
                FROM plate_reads
                WHERE {' AND '.join(clauses)}
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
                """,
                params,
            )
            return [_read_row_to_dict(row) for row in cur.fetchall()]


def search_plates(query: str, *, limit: int = 25) -> dict:
    normalized = normalize_plate_text(query)
    return {
        "lists": [
            entry
            for entry in list_plate_entries()
            if normalized in entry["normalizedPlate"]
        ][:limit],
        "reads": query_reads(plate=query, limit=limit),
    }


def _validate_plate_entry(plate_text: str, list_type: str) -> str:
    if list_type not in LIST_TYPES:
        raise ValueError("Unsupported plate list type")
    normalized = normalize_plate_text(plate_text)
    if not normalized:
        raise ValueError("Plate number is required")
    if len(normalized) < 4:
        raise ValueError("Plate number is too short")
    return normalized


def _normalize_list_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"visitor", "visitors"}:
        return "visitors"
    if normalized in {"blocklist", "blacklist", "blocked"}:
        return "blocked"
    if normalized in {"allowlist", "whitelist"}:
        return "whitelist"
    return normalized


def _is_ocr_equivalent(left: str, right: str) -> bool:
    return any(left in group and right in group for group in OCR_EQUIVALENT_GROUPS)


def _save_runtime_file(root: Path, file_id: str, blob: bytes | None) -> str | None:
    if not blob:
        return None
    root.mkdir(exist_ok=True)
    filename = f"{file_id}.jpg"
    (root / filename).write_bytes(blob)
    return filename


def _plate_row_to_dict(row: dict) -> dict:
    return {
        "id": row["id"],
        "plateNumber": row["plate_text"],
        "normalizedPlate": row["normalized_plate"],
        "list": row["list_type"],
        "owner": row.get("owner_name") or "",
        "vehicle": row.get("vehicle_desc") or "",
        "validFrom": row.get("valid_from"),
        "validUntil": row.get("valid_until"),
        "createdAt": row["created_at"],
        "active": row.get("active", True),
        "isIndianFormat": is_indian_plate_format(row["normalized_plate"]),
    }


def _read_row_to_dict(row: dict) -> dict:
    bbox = row.get("bbox")
    if isinstance(bbox, str):
        bbox = json.loads(bbox)
    return {
        "id": row["id"],
        "plateNumber": row["plate_text"],
        "normalizedPlate": row["normalized_plate"],
        "cameraId": row["camera_id"],
        "cameraName": row["camera_name"],
        "timestamp": row["timestamp"],
        "eventType": row["event_type"],
        "matchStatus": _event_to_match_status(row["event_type"], row.get("matched_list")),
        "matchedListId": row.get("matched_list_id"),
        "matchedList": row.get("matched_list"),
        "confidence": row.get("confidence"),
        "detectionConfidence": row.get("detection_confidence"),
        "ocrConfidence": row.get("ocr_confidence"),
        "snapshotUrl": f"/api/plates/reads/{row['id']}/snapshot" if row.get("snapshot_path") else None,
        "cropUrl": f"/api/plates/reads/{row['id']}/crop" if row.get("crop_path") else None,
        "bbox": bbox or {},
        "vehicleClass": row.get("vehicle_class"),
        "qualityReason": row.get("quality_reason"),
        "isUnknown": row["event_type"] == "plate_unknown",
    }


def _event_to_match_status(event_type: str, matched_list: str | None) -> str:
    if event_type == "plate_blocked":
        return "blocked"
    if event_type == "plate_visitor":
        return "visitor"
    if matched_list == "whitelist":
        return "whitelist"
    if event_type == "plate_low_confidence":
        return "low_confidence"
    return "unknown"
