"""
Rakshak Lens edge licensing — verify signed license files and heartbeat tokens,
compute current license state, and persist uploaded licenses to disk.

This module is intentionally side-effect free at import time. Call
``init_licensing()`` from FastAPI startup before any other entry point.

The full spec lives at ``docs/licensing-spec.md``. Both this module and the
License Hub project must canonicalize and sign payloads identically.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from runtime_storage import STATE_DIR, atomic_write_private

logger = logging.getLogger("rakshak_lens.licensing")

# ── Paths ───────────────────────────────────────────────────────────────────

_BACKEND_DIR = Path(__file__).parent
PUBLIC_KEY_PATH = _BACKEND_DIR / "keys" / "license_pub.pem"
LICENSE_DIR = STATE_DIR / "license"
HEARTBEAT_DIR = STATE_DIR / "heartbeat"
LICENSE_PATH = LICENSE_DIR / "current.lic"
HEARTBEAT_PATH = HEARTBEAT_DIR / "current.json"

# ── Policy constants ────────────────────────────────────────────────────────

SCHEMA_VERSION = 1
GRACE_PERIOD_DAYS = 14
WARNING_THRESHOLD_DAYS = 14
HEARTBEAT_VALIDITY_DAYS = 35  # set by License Hub when issuing; documented here for reference

# Heartbeat refresh schedule. Configurable via env so tests can run with a
# tight loop without changing code.
HEARTBEAT_REFRESH_INTERVAL_SECONDS = int(os.environ.get("LICENSE_HEARTBEAT_REFRESH_SECONDS", 24 * 60 * 60))
HEARTBEAT_REFRESH_INITIAL_DELAY_SECONDS = int(
    os.environ.get("LICENSE_HEARTBEAT_INITIAL_DELAY_SECONDS", 60)
)
HEARTBEAT_REFRESH_HTTP_TIMEOUT = 30

# ── Errors ──────────────────────────────────────────────────────────────────


class LicensingError(Exception):
    """Base class for all licensing errors."""


class InvalidLicense(LicensingError):
    """The license file is malformed, has a bad signature, or wrong schema version."""


class InvalidHeartbeat(LicensingError):
    """The heartbeat token is malformed, has a bad signature, or wrong schema version."""


# ── Enums and dataclasses ───────────────────────────────────────────────────


class LicenseState(str, Enum):
    """The four enforcement states the edge can be in."""

    VALID = "valid"
    WARNING = "warning"
    GRACE = "grace"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class License:
    schema_version: int
    license_id: str
    customer_name: str
    issued_by_partner: str
    max_cameras: int
    features: tuple[str, ...]
    issued_at: datetime
    expires_at: datetime
    heartbeat_url: str

    def to_public_dict(self) -> dict:
        """Serializable shape for the API — no signature, ISO timestamps."""
        return {
            "schema_version": self.schema_version,
            "license_id": self.license_id,
            "customer_name": self.customer_name,
            "issued_by_partner": self.issued_by_partner,
            "max_cameras": self.max_cameras,
            "features": list(self.features),
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "heartbeat_url": self.heartbeat_url,
        }


@dataclass(frozen=True)
class Heartbeat:
    schema_version: int
    license_id: str
    issued_at: datetime
    valid_until: datetime

    def to_public_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "license_id": self.license_id,
            "issued_at": self.issued_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
        }


@dataclass(frozen=True)
class LicenseStatus:
    """Computed at startup, on every upload, and periodically."""

    state: LicenseState
    license: Optional[License]
    heartbeat: Optional[Heartbeat]
    reason: str
    days_until_suspension: Optional[int]
    has_license: bool

    def to_public_dict(self) -> dict:
        return {
            "state": self.state.value,
            "license": self.license.to_public_dict() if self.license else None,
            "heartbeat": self.heartbeat.to_public_dict() if self.heartbeat else None,
            "reason": self.reason,
            "days_until_suspension": self.days_until_suspension,
            "has_license": self.has_license,
            "inference_allowed": self.state != LicenseState.SUSPENDED,
        }


# ── Module state (thread-safe) ──────────────────────────────────────────────

_lock = threading.RLock()
_public_key: Optional[Ed25519PublicKey] = None
_cached_status: Optional[LicenseStatus] = None


# ── Canonicalization ────────────────────────────────────────────────────────


def _canonicalize(payload: dict) -> bytes:
    """Canonical JSON encoding used for both signing and verification.

    Both License Hub and edge MUST produce byte-identical output for the same
    logical payload, otherwise signatures won't match. Sort keys, no spaces,
    no ASCII escaping.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 timestamp into a tz-aware UTC datetime."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── Verification ────────────────────────────────────────────────────────────


def _load_public_key() -> Ed25519PublicKey:
    if not PUBLIC_KEY_PATH.is_file():
        raise LicensingError(
            f"Public key not found at {PUBLIC_KEY_PATH}. "
            "This is a build error — the public key must be bundled with the image."
        )
    pem = PUBLIC_KEY_PATH.read_bytes()
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise LicensingError(
            f"Expected Ed25519 public key at {PUBLIC_KEY_PATH}, got {type(key).__name__}"
        )
    return key


def verify_license(blob: bytes, public_key: Optional[Ed25519PublicKey] = None) -> License:
    """Verify a raw license file blob and return a parsed License object.

    Raises InvalidLicense on any failure: bad JSON, missing fields, wrong
    schema version, bad signature, missing required fields.
    """
    pk = public_key or _public_key
    if pk is None:
        raise LicensingError("Public key not loaded — did you call init_licensing()?")

    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InvalidLicense(f"License is not valid UTF-8 JSON: {e}") from e

    if not isinstance(payload, dict):
        raise InvalidLicense("License must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InvalidLicense(
            f"Unsupported schema_version {payload.get('schema_version')}, expected {SCHEMA_VERSION}"
        )

    sig_b64 = payload.pop("signature", None)
    if not sig_b64:
        raise InvalidLicense("License is missing 'signature' field")
    try:
        sig = base64.b64decode(sig_b64, validate=True)
    except Exception as e:
        raise InvalidLicense(f"License signature is not valid base64: {e}") from e

    canonical = _canonicalize(payload)
    try:
        pk.verify(sig, canonical)
    except InvalidSignature as e:
        raise InvalidLicense("License signature verification failed") from e

    required = (
        "license_id",
        "customer_name",
        "issued_by_partner",
        "max_cameras",
        "features",
        "issued_at",
        "expires_at",
        "heartbeat_url",
    )
    for field in required:
        if field not in payload:
            raise InvalidLicense(f"License is missing required field '{field}'")

    if not isinstance(payload["max_cameras"], int) or payload["max_cameras"] <= 0:
        raise InvalidLicense("max_cameras must be a positive integer")
    if not isinstance(payload["features"], list) or not all(
        isinstance(f, str) for f in payload["features"]
    ):
        raise InvalidLicense("features must be a list of strings")

    return License(
        schema_version=payload["schema_version"],
        license_id=payload["license_id"],
        customer_name=payload["customer_name"],
        issued_by_partner=payload["issued_by_partner"],
        max_cameras=payload["max_cameras"],
        features=tuple(payload["features"]),
        issued_at=_parse_iso(payload["issued_at"]),
        expires_at=_parse_iso(payload["expires_at"]),
        heartbeat_url=payload["heartbeat_url"],
    )


def verify_heartbeat(
    blob: bytes,
    license_id: str,
    public_key: Optional[Ed25519PublicKey] = None,
) -> Heartbeat:
    """Verify a heartbeat token and return a parsed Heartbeat.

    The heartbeat must reference the given ``license_id``; this prevents a
    heartbeat from one license from being replayed against a different one.
    """
    pk = public_key or _public_key
    if pk is None:
        raise LicensingError("Public key not loaded — did you call init_licensing()?")

    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InvalidHeartbeat(f"Heartbeat is not valid UTF-8 JSON: {e}") from e

    if not isinstance(payload, dict):
        raise InvalidHeartbeat("Heartbeat must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InvalidHeartbeat(
            f"Unsupported heartbeat schema_version {payload.get('schema_version')}"
        )

    sig_b64 = payload.pop("signature", None)
    if not sig_b64:
        raise InvalidHeartbeat("Heartbeat is missing 'signature' field")
    try:
        sig = base64.b64decode(sig_b64, validate=True)
    except Exception as e:
        raise InvalidHeartbeat(f"Heartbeat signature is not valid base64: {e}") from e

    canonical = _canonicalize(payload)
    try:
        pk.verify(sig, canonical)
    except InvalidSignature as e:
        raise InvalidHeartbeat("Heartbeat signature verification failed") from e

    for field in ("license_id", "issued_at", "valid_until"):
        if field not in payload:
            raise InvalidHeartbeat(f"Heartbeat is missing required field '{field}'")

    if payload["license_id"] != license_id:
        raise InvalidHeartbeat(
            f"Heartbeat license_id '{payload['license_id']}' does not match installed license '{license_id}'"
        )

    return Heartbeat(
        schema_version=payload["schema_version"],
        license_id=payload["license_id"],
        issued_at=_parse_iso(payload["issued_at"]),
        valid_until=_parse_iso(payload["valid_until"]),
    )


# ── State machine ───────────────────────────────────────────────────────────


def compute_status(
    license: Optional[License],
    heartbeat: Optional[Heartbeat],
    now: Optional[datetime] = None,
) -> LicenseStatus:
    """Pure function: given a license, heartbeat, and current time, return status.

    The state machine considers BOTH the license expiry and the heartbeat
    expiry. The "worse" of the two wins — if either is past grace, we suspend.
    """
    now = now or datetime.now(timezone.utc)

    if license is None:
        return LicenseStatus(
            state=LicenseState.SUSPENDED,
            license=None,
            heartbeat=None,
            reason="No license installed. Please upload a .lic file from your implementation partner.",
            days_until_suspension=0,
            has_license=False,
        )

    license_expired = now >= license.expires_at
    license_grace_end = license.expires_at + timedelta(days=GRACE_PERIOD_DAYS)
    license_past_grace = now >= license_grace_end
    license_warning_threshold = license.expires_at - timedelta(days=WARNING_THRESHOLD_DAYS)
    license_in_warning_window = now >= license_warning_threshold

    # A missing heartbeat is only healthy during the first signed heartbeat
    # window. Deriving that window from license.issued_at prevents deleting the
    # heartbeat file (or recreating a container) from resetting revocation.
    hb_valid_until = (
        heartbeat.valid_until
        if heartbeat is not None
        else license.issued_at + timedelta(days=HEARTBEAT_VALIDITY_DAYS)
    )
    hb_past_validity = now >= hb_valid_until
    hb_grace_end = hb_valid_until + timedelta(days=GRACE_PERIOD_DAYS)
    hb_past_grace = hb_past_validity and now >= hb_grace_end
    suspension_at = min(license_grace_end, hb_grace_end)

    # Suspension takes priority
    if license_past_grace:
        return LicenseStatus(
            state=LicenseState.SUSPENDED,
            license=license,
            heartbeat=heartbeat,
            reason=(
                f"License expired on {license.expires_at.date()} and the "
                f"{GRACE_PERIOD_DAYS}-day grace period has passed. Inference is suspended."
            ),
            days_until_suspension=0,
            has_license=True,
        )
    if hb_past_grace:
        last_seen = heartbeat.valid_until.date() if heartbeat else hb_valid_until.date()
        return LicenseStatus(
            state=LicenseState.SUSPENDED,
            license=license,
            heartbeat=heartbeat,
            reason=(
                f"License heartbeat last refreshed on {last_seen} and the "
                f"{GRACE_PERIOD_DAYS}-day grace period has passed. "
                "Check internet connectivity or upload a fresh heartbeat token."
            ),
            days_until_suspension=0,
            has_license=True,
        )

    # Then grace period (license/heartbeat past validity but within grace)
    if license_expired:
        days_left = max(0, (suspension_at - now).days)
        return LicenseStatus(
            state=LicenseState.GRACE,
            license=license,
            heartbeat=heartbeat,
            reason=(
                f"License expired on {license.expires_at.date()}. "
                f"Inference will be suspended in {days_left} days. Please renew."
            ),
            days_until_suspension=days_left,
            has_license=True,
        )
    if hb_past_validity:
        days_left = max(0, (suspension_at - now).days)
        return LicenseStatus(
            state=LicenseState.GRACE,
            license=license,
            heartbeat=heartbeat,
            reason=(
                "License heartbeat refresh is overdue. "
                f"Inference will be suspended in {days_left} days if not refreshed."
            ),
            days_until_suspension=days_left,
            has_license=True,
        )

    # Then warning (close to expiry but still healthy)
    if license_in_warning_window:
        days_left = max(0, (license.expires_at - now).days)
        days_until_suspension = max(0, (suspension_at - now).days)
        return LicenseStatus(
            state=LicenseState.WARNING,
            license=license,
            heartbeat=heartbeat,
            reason=f"License expires in {days_left} days on {license.expires_at.date()}. Please plan to renew.",
            days_until_suspension=days_until_suspension,
            has_license=True,
        )

    # All good
    days_left = max(0, (suspension_at - now).days)
    return LicenseStatus(
        state=LicenseState.VALID,
        license=license,
        heartbeat=heartbeat,
        reason="License is valid.",
        days_until_suspension=days_left,
        has_license=True,
    )


# ── Persistence ─────────────────────────────────────────────────────────────


def _atomic_write(path: Path, content: bytes) -> None:
    atomic_write_private(path, content)


def _load_license_from_disk() -> Optional[License]:
    if not LICENSE_PATH.is_file():
        return None
    try:
        return verify_license(LICENSE_PATH.read_bytes())
    except InvalidLicense as e:
        logger.error("Stored license at %s failed verification: %s", LICENSE_PATH, e)
        return None


def _load_heartbeat_from_disk(
    license_id: str,
    *,
    license_issued_at: datetime | None = None,
) -> Optional[Heartbeat]:
    if not HEARTBEAT_PATH.is_file():
        return None
    try:
        heartbeat = verify_heartbeat(HEARTBEAT_PATH.read_bytes(), license_id)
    except InvalidHeartbeat as e:
        logger.error("Stored heartbeat at %s failed verification: %s", HEARTBEAT_PATH, e)
        return None
    if license_issued_at is not None and heartbeat.valid_until < license_issued_at:
        logger.info(
            "Ignoring stale heartbeat for %s: heartbeat valid_until=%s is before license issued_at=%s",
            license_id,
            heartbeat.valid_until.isoformat(),
            license_issued_at.isoformat(),
        )
        return None
    return heartbeat


def save_license(blob: bytes) -> License:
    """Verify and persist an uploaded license blob. Updates cached status.

    Raises InvalidLicense on verification failure — the file is not written
    if verification fails, so a bad upload can never replace a good license.
    """
    with _lock:
        license = verify_license(blob)
        _atomic_write(LICENSE_PATH, blob)
        # Reload heartbeat against the new license_id; if the old heartbeat
        # was for a different license, it will be ignored.
        heartbeat = _load_heartbeat_from_disk(
            license.license_id,
            license_issued_at=license.issued_at,
        )
        global _cached_status
        _cached_status = compute_status(license, heartbeat)
        logger.info(
            "License installed: %s for %s, expires %s",
            license.license_id,
            license.customer_name,
            license.expires_at.date(),
        )
        return license


def save_heartbeat(blob: bytes) -> Heartbeat:
    """Verify and persist an uploaded heartbeat token blob. Updates cached status."""
    with _lock:
        license = _load_license_from_disk()
        if license is None:
            raise InvalidHeartbeat("Cannot install heartbeat without an installed license")
        heartbeat = verify_heartbeat(blob, license.license_id)
        _atomic_write(HEARTBEAT_PATH, blob)
        global _cached_status
        _cached_status = compute_status(license, heartbeat)
        logger.info(
            "Heartbeat installed for %s, valid until %s",
            heartbeat.license_id,
            heartbeat.valid_until.date(),
        )
        return heartbeat


# ── Public API ──────────────────────────────────────────────────────────────


def init_licensing() -> LicenseStatus:
    """Load public key, read any existing license/heartbeat from disk, and
    compute the initial status. Call once at FastAPI startup.
    """
    global _public_key, _cached_status
    with _lock:
        for directory in (LICENSE_DIR, HEARTBEAT_DIR):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
        for state_file in (LICENSE_PATH, HEARTBEAT_PATH):
            if state_file.is_file():
                os.chmod(state_file, 0o600)
        _public_key = _load_public_key()
        license = _load_license_from_disk()
        heartbeat = (
            _load_heartbeat_from_disk(license.license_id, license_issued_at=license.issued_at)
            if license
            else None
        )
        _cached_status = compute_status(license, heartbeat)
        logger.info(
            "Licensing initialized: state=%s reason=%s",
            _cached_status.state.value,
            _cached_status.reason,
        )
        return _cached_status


def get_status() -> LicenseStatus:
    """Return the current license status, recomputing against the wall clock.

    Recomputing on every call lets transitions (warning → grace → suspended)
    happen automatically as time passes, without needing a separate poller.
    """
    global _cached_status
    with _lock:
        if _cached_status is None:
            return init_licensing()
        fresh = compute_status(_cached_status.license, _cached_status.heartbeat)
        if fresh.state != _cached_status.state:
            logger.info(
                "License state transition: %s → %s (%s)",
                _cached_status.state.value,
                fresh.state.value,
                fresh.reason,
            )
        _cached_status = fresh
        return fresh


def is_inference_allowed() -> bool:
    """Convenience check used by inference workers."""
    return get_status().state != LicenseState.SUSPENDED


def can_add_camera(current_camera_count: int) -> tuple[bool, str]:
    """Check whether the configured camera count is below the licensed cap.

    Returns ``(allowed, reason)``. ``reason`` is empty when allowed.
    """
    status = get_status()
    if status.license is None:
        return False, "No license installed"
    if current_camera_count >= status.license.max_cameras:
        return (
            False,
            f"License allows up to {status.license.max_cameras} cameras. "
            f"Contact your implementation partner to upgrade.",
        )
    return True, ""


def has_feature(feature: str) -> bool:
    """Check whether a feature flag is licensed."""
    status = get_status()
    if status.license is None:
        return False
    return feature in status.license.features


# ── Heartbeat refresh loop ──────────────────────────────────────────────────


def _attempt_heartbeat_refresh() -> bool:
    """One-shot heartbeat refresh. Returns True on success.

    Reads the current license, POSTs to its heartbeat_url, validates the
    response signature, and persists. Any failure is logged and ignored —
    the existing heartbeat continues to be valid until its grace period
    runs out.
    """
    status = get_status()
    if status.license is None:
        logger.debug("Skipping heartbeat refresh — no license installed")
        return False

    url = status.license.heartbeat_url
    license_id = status.license.license_id
    try:
        response = requests.post(
            url,
            json={"license_id": license_id},
            timeout=HEARTBEAT_REFRESH_HTTP_TIMEOUT,
            headers={"User-Agent": "Rakshak Lens-Edge/1.0"},
        )
    except requests.RequestException as e:
        logger.warning("Heartbeat refresh network error for %s: %s", license_id, e)
        return False

    if response.status_code != 200:
        # 403 likely means revoked or expired — log loud so it's visible.
        log_fn = logger.error if response.status_code in (403, 410) else logger.warning
        log_fn(
            "Heartbeat refresh rejected by hub: %d %s",
            response.status_code,
            response.text[:200],
        )
        return False

    try:
        save_heartbeat(response.content)
    except InvalidHeartbeat as e:
        logger.error("Hub returned a heartbeat that failed verification: %s", e)
        return False

    logger.info("Heartbeat refreshed successfully for %s", license_id)
    return True


async def heartbeat_refresh_loop() -> None:
    """Background task that periodically refreshes the heartbeat token.

    Run as an asyncio task from FastAPI startup. Sleeps 24h between
    attempts (with jitter on first run so a fleet of edges doesn't hammer
    the hub all at once). Network and signing happen in a thread to avoid
    blocking the event loop.
    """
    # Wait briefly so the first attempt doesn't block startup or race with
    # license initialization. Add jitter so multiple edge devices booting
    # together don't all hit the hub on the same second.
    initial_jitter = random.uniform(0, 30)
    await asyncio.sleep(HEARTBEAT_REFRESH_INITIAL_DELAY_SECONDS + initial_jitter)

    while True:
        try:
            await asyncio.to_thread(_attempt_heartbeat_refresh)
        except Exception:
            # to_thread shouldn't raise — _attempt_heartbeat_refresh swallows
            # everything — but be defensive so a bug here doesn't kill the loop.
            logger.exception("Unexpected error in heartbeat refresh loop")

        # Wait a day with small jitter before next attempt.
        jitter = random.uniform(0, 600)
        await asyncio.sleep(HEARTBEAT_REFRESH_INTERVAL_SECONDS + jitter)
