"""Tests for the licensing module — signature verification and state machine."""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Add backend to import path so `import licensing` works the same way the
# rest of the backend tests do.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import licensing


# ── Test fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def test_keypair():
    """Generate a fresh Ed25519 keypair just for this test."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def configured_module(test_keypair, tmp_path, monkeypatch):
    """Replace module-level paths and public key with test values."""
    private_key, public_key = test_keypair

    license_dir = tmp_path / "license"
    heartbeat_dir = tmp_path / "heartbeat"
    license_path = license_dir / "current.lic"
    heartbeat_path = heartbeat_dir / "current.json"

    monkeypatch.setattr(licensing, "LICENSE_DIR", license_dir)
    monkeypatch.setattr(licensing, "HEARTBEAT_DIR", heartbeat_dir)
    monkeypatch.setattr(licensing, "LICENSE_PATH", license_path)
    monkeypatch.setattr(licensing, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(licensing, "_public_key", public_key)
    monkeypatch.setattr(licensing, "_cached_status", None)
    # init_licensing() reloads the public key from disk; for tests we want to
    # keep using our test keypair instead of the bundled production one.
    monkeypatch.setattr(licensing, "_load_public_key", lambda: public_key)

    return private_key, public_key


def _sign_license(private_key: Ed25519PrivateKey, **overrides) -> bytes:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = {
        "schema_version": 1,
        "license_id": "SL-2026-9999",
        "customer_name": "Test Customer",
        "issued_by_partner": "techser",
        "max_cameras": 10,
        "features": ["base"],
        "issued_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(days=365)).isoformat().replace("+00:00", "Z"),
        "heartbeat_url": "https://licenses.techser.com/api/heartbeat",
    }
    payload.update(overrides)
    canonical = licensing._canonicalize(payload)
    sig = private_key.sign(canonical)
    return json.dumps({**payload, "signature": base64.b64encode(sig).decode("ascii")}).encode()


def _sign_heartbeat(private_key: Ed25519PrivateKey, **overrides) -> bytes:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = {
        "schema_version": 1,
        "license_id": "SL-2026-9999",
        "issued_at": now.isoformat().replace("+00:00", "Z"),
        "valid_until": (now + timedelta(days=35)).isoformat().replace("+00:00", "Z"),
    }
    payload.update(overrides)
    canonical = licensing._canonicalize(payload)
    sig = private_key.sign(canonical)
    return json.dumps({**payload, "signature": base64.b64encode(sig).decode("ascii")}).encode()


# ── verify_license ──────────────────────────────────────────────────────────


class TestVerifyLicense:
    def test_valid_license_roundtrip(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_license(private_key)
        license = licensing.verify_license(blob, public_key)
        assert license.license_id == "SL-2026-9999"
        assert license.max_cameras == 10
        assert license.features == ("base",)
        assert license.customer_name == "Test Customer"

    def test_tampered_payload_fails(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_license(private_key, max_cameras=10)
        # Flip max_cameras after signing
        payload = json.loads(blob.decode())
        payload["max_cameras"] = 9999
        tampered = json.dumps(payload).encode()
        with pytest.raises(licensing.InvalidLicense, match="signature verification failed"):
            licensing.verify_license(tampered, public_key)

    def test_wrong_public_key_fails(self, configured_module):
        private_key, _ = configured_module
        wrong_key = Ed25519PrivateKey.generate().public_key()
        blob = _sign_license(private_key)
        with pytest.raises(licensing.InvalidLicense, match="signature verification failed"):
            licensing.verify_license(blob, wrong_key)

    def test_missing_signature_fails(self, configured_module):
        _, public_key = configured_module
        payload = {
            "schema_version": 1,
            "license_id": "SL-X",
            "customer_name": "X",
            "issued_by_partner": "x",
            "max_cameras": 1,
            "features": ["base"],
            "issued_at": "2026-01-01T00:00:00Z",
            "expires_at": "2027-01-01T00:00:00Z",
            "heartbeat_url": "https://x",
        }
        with pytest.raises(licensing.InvalidLicense, match="missing 'signature'"):
            licensing.verify_license(json.dumps(payload).encode(), public_key)

    def test_unsupported_schema_version_fails(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_license(private_key, schema_version=99)
        with pytest.raises(licensing.InvalidLicense, match="schema_version"):
            licensing.verify_license(blob, public_key)

    def test_invalid_json_fails(self, configured_module):
        _, public_key = configured_module
        with pytest.raises(licensing.InvalidLicense, match="not valid"):
            licensing.verify_license(b"not json {", public_key)

    def test_negative_max_cameras_fails(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_license(private_key, max_cameras=-1)
        with pytest.raises(licensing.InvalidLicense, match="max_cameras"):
            licensing.verify_license(blob, public_key)

    def test_features_must_be_strings(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_license(private_key, features=[1, 2, 3])
        with pytest.raises(licensing.InvalidLicense, match="features"):
            licensing.verify_license(blob, public_key)


# ── verify_heartbeat ────────────────────────────────────────────────────────


class TestVerifyHeartbeat:
    def test_valid_heartbeat_roundtrip(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_heartbeat(private_key)
        hb = licensing.verify_heartbeat(blob, "SL-2026-9999", public_key)
        assert hb.license_id == "SL-2026-9999"

    def test_mismatched_license_id_fails(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_heartbeat(private_key, license_id="SL-DIFFERENT")
        with pytest.raises(licensing.InvalidHeartbeat, match="does not match"):
            licensing.verify_heartbeat(blob, "SL-2026-9999", public_key)

    def test_tampered_heartbeat_fails(self, configured_module):
        private_key, public_key = configured_module
        blob = _sign_heartbeat(private_key)
        payload = json.loads(blob.decode())
        payload["valid_until"] = "2099-01-01T00:00:00Z"
        with pytest.raises(licensing.InvalidHeartbeat, match="signature verification failed"):
            licensing.verify_heartbeat(json.dumps(payload).encode(), "SL-2026-9999", public_key)


# ── State machine ──────────────────────────────────────────────────────────


def _license(expires_at: datetime, **kwargs) -> licensing.License:
    defaults = {
        "schema_version": 1,
        "license_id": "SL-2026-0001",
        "customer_name": "Acme",
        "issued_by_partner": "techser",
        "max_cameras": 10,
        "features": ("base",),
        "issued_at": expires_at - timedelta(days=365),
        "expires_at": expires_at,
        "heartbeat_url": "https://licenses.techser.com/api/heartbeat",
    }
    defaults.update(kwargs)
    return licensing.License(**defaults)


def _heartbeat(valid_until: datetime, **kwargs) -> licensing.Heartbeat:
    defaults = {
        "schema_version": 1,
        "license_id": "SL-2026-0001",
        "issued_at": valid_until - timedelta(days=35),
        "valid_until": valid_until,
    }
    defaults.update(kwargs)
    return licensing.Heartbeat(**defaults)


class TestComputeStatus:
    def test_no_license_is_suspended(self):
        status = licensing.compute_status(None, None)
        assert status.state == licensing.LicenseState.SUSPENDED
        assert not status.has_license
        assert "No license" in status.reason

    def test_fresh_license_with_fresh_heartbeat_is_valid(self):
        now = datetime.now(timezone.utc)
        status = licensing.compute_status(
            _license(now + timedelta(days=300)),
            _heartbeat(now + timedelta(days=30)),
            now=now,
        )
        assert status.state == licensing.LicenseState.VALID
        assert status.days_until_suspension > 300

    def test_license_within_warning_window_warns(self):
        now = datetime.now(timezone.utc)
        # 7 days from expiry — inside the 14-day warning window
        status = licensing.compute_status(
            _license(now + timedelta(days=7)),
            _heartbeat(now + timedelta(days=30)),
            now=now,
        )
        assert status.state == licensing.LicenseState.WARNING

    def test_expired_license_within_grace_is_grace(self):
        now = datetime.now(timezone.utc)
        # Expired 5 days ago — still within 14-day grace
        status = licensing.compute_status(
            _license(now - timedelta(days=5)),
            _heartbeat(now + timedelta(days=30)),
            now=now,
        )
        assert status.state == licensing.LicenseState.GRACE
        assert status.days_until_suspension == 9  # 14-day grace, 5 days elapsed

    def test_expired_license_past_grace_is_suspended(self):
        now = datetime.now(timezone.utc)
        status = licensing.compute_status(
            _license(now - timedelta(days=20)),
            _heartbeat(now + timedelta(days=30)),
            now=now,
        )
        assert status.state == licensing.LicenseState.SUSPENDED
        assert status.days_until_suspension == 0

    def test_no_heartbeat_with_fresh_license_is_valid(self):
        now = datetime.now(timezone.utc)
        # Brand-new installs should not be penalized before the first heartbeat.
        status = licensing.compute_status(
            _license(now + timedelta(days=300)),
            None,
            now=now,
        )
        assert status.state == licensing.LicenseState.VALID

    def test_expired_heartbeat_within_grace_is_grace(self):
        now = datetime.now(timezone.utc)
        # Heartbeat expired 7 days ago, license still healthy
        status = licensing.compute_status(
            _license(now + timedelta(days=300)),
            _heartbeat(now - timedelta(days=7)),
            now=now,
        )
        assert status.state == licensing.LicenseState.GRACE

    def test_expired_heartbeat_past_grace_is_suspended(self):
        now = datetime.now(timezone.utc)
        status = licensing.compute_status(
            _license(now + timedelta(days=300)),
            _heartbeat(now - timedelta(days=20)),
            now=now,
        )
        assert status.state == licensing.LicenseState.SUSPENDED
        assert "heartbeat" in status.reason.lower()

    def test_suspended_takes_priority_over_warning(self):
        now = datetime.now(timezone.utc)
        # License is in warning window AND heartbeat is past grace
        status = licensing.compute_status(
            _license(now + timedelta(days=7)),
            _heartbeat(now - timedelta(days=20)),
            now=now,
        )
        assert status.state == licensing.LicenseState.SUSPENDED


# ── Persistence ─────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_license_persists_to_disk(self, configured_module):
        private_key, _ = configured_module
        blob = _sign_license(private_key)
        license = licensing.save_license(blob)
        assert licensing.LICENSE_PATH.exists()
        assert license.license_id == "SL-2026-9999"
        assert licensing.LICENSE_PATH.read_bytes() == blob

    def test_save_invalid_license_does_not_overwrite(self, configured_module):
        private_key, _ = configured_module
        good_blob = _sign_license(private_key)
        licensing.save_license(good_blob)
        bad_blob = b"not a license"
        with pytest.raises(licensing.InvalidLicense):
            licensing.save_license(bad_blob)
        # File should still be the good one
        assert licensing.LICENSE_PATH.read_bytes() == good_blob

    def test_save_heartbeat_requires_license(self, configured_module):
        private_key, _ = configured_module
        blob = _sign_heartbeat(private_key)
        with pytest.raises(licensing.InvalidHeartbeat, match="without an installed license"):
            licensing.save_heartbeat(blob)

    def test_save_heartbeat_with_matching_license(self, configured_module):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key))
        hb = licensing.save_heartbeat(_sign_heartbeat(private_key))
        assert hb.license_id == "SL-2026-9999"
        assert licensing.HEARTBEAT_PATH.exists()

    def test_save_heartbeat_for_wrong_license_fails(self, configured_module):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key, license_id="SL-2026-AAAA"))
        wrong_hb = _sign_heartbeat(private_key, license_id="SL-2026-BBBB")
        with pytest.raises(licensing.InvalidHeartbeat, match="does not match"):
            licensing.save_heartbeat(wrong_hb)

    def test_save_renewed_license_ignores_heartbeat_expired_before_issue(self, configured_module):
        private_key, _ = configured_module
        now = datetime.now(timezone.utc).replace(microsecond=0)
        stale_valid_until = now - timedelta(days=1)

        licensing.HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        licensing.HEARTBEAT_PATH.write_bytes(
            _sign_heartbeat(
                private_key,
                valid_until=stale_valid_until.isoformat().replace("+00:00", "Z"),
                issued_at=(stale_valid_until - timedelta(days=35)).isoformat().replace("+00:00", "Z"),
            )
        )

        licensing.save_license(
            _sign_license(
                private_key,
                issued_at=now.isoformat().replace("+00:00", "Z"),
                expires_at=(now + timedelta(days=365)).isoformat().replace("+00:00", "Z"),
            )
        )

        status = licensing.get_status()
        assert status.state == licensing.LicenseState.VALID
        assert status.heartbeat is None


# ── Public API ──────────────────────────────────────────────────────────────


class TestPublicAPI:
    def test_get_status_with_no_license(self, configured_module):
        status = licensing.get_status()
        assert status.state == licensing.LicenseState.SUSPENDED
        assert not licensing.is_inference_allowed()

    def test_get_status_after_install(self, configured_module):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key))
        licensing.save_heartbeat(_sign_heartbeat(private_key))
        status = licensing.get_status()
        assert status.state == licensing.LicenseState.VALID
        assert licensing.is_inference_allowed()

    def test_can_add_camera_below_limit(self, configured_module):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key, max_cameras=10))
        licensing.save_heartbeat(_sign_heartbeat(private_key))
        allowed, reason = licensing.can_add_camera(5)
        assert allowed
        assert reason == ""

    def test_can_add_camera_at_limit_blocked(self, configured_module):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key, max_cameras=10))
        licensing.save_heartbeat(_sign_heartbeat(private_key))
        allowed, reason = licensing.can_add_camera(10)
        assert not allowed
        assert "10 cameras" in reason

    def test_can_add_camera_no_license(self, configured_module):
        allowed, reason = licensing.can_add_camera(0)
        assert not allowed
        assert "No license" in reason

    def test_has_feature_present(self, configured_module):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key, features=["base", "anpr"]))
        licensing.save_heartbeat(_sign_heartbeat(private_key))
        assert licensing.has_feature("base")
        assert licensing.has_feature("anpr")

    def test_has_feature_absent(self, configured_module):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key, features=["base"]))
        licensing.save_heartbeat(_sign_heartbeat(private_key))
        assert not licensing.has_feature("anpr")

    def test_has_feature_no_license(self, configured_module):
        assert not licensing.has_feature("base")

    def test_init_licensing_loads_existing_files(self, configured_module, tmp_path):
        private_key, _ = configured_module
        # Pre-write license and heartbeat to disk before init
        licensing.LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        licensing.HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        licensing.LICENSE_PATH.write_bytes(_sign_license(private_key))
        licensing.HEARTBEAT_PATH.write_bytes(_sign_heartbeat(private_key))
        # Reset cache and re-init
        licensing._cached_status = None
        status = licensing.init_licensing()
        assert status.state == licensing.LicenseState.VALID
        assert status.license is not None
        assert status.heartbeat is not None


# ── Heartbeat refresh ───────────────────────────────────────────────────────


class TestHeartbeatRefresh:
    def test_refresh_succeeds_when_hub_returns_valid_token(self, configured_module, monkeypatch):
        private_key, _ = configured_module
        # Install a license so the refresh has something to refresh against
        licensing.save_license(_sign_license(private_key))

        # Mock requests.post to return a fresh heartbeat
        fresh_heartbeat = _sign_heartbeat(private_key)

        class FakeResponse:
            status_code = 200
            content = fresh_heartbeat
            text = fresh_heartbeat.decode()

        def fake_post(url, json, timeout, headers):
            assert json["license_id"] == "SL-2026-9999"
            return FakeResponse()

        monkeypatch.setattr(licensing.requests, "post", fake_post)

        result = licensing._attempt_heartbeat_refresh()
        assert result is True
        assert licensing.HEARTBEAT_PATH.exists()
        # The new heartbeat is now the cached one
        status = licensing.get_status()
        assert status.heartbeat is not None

    def test_refresh_no_license_skips(self, configured_module, monkeypatch):
        called = []

        def fake_post(*args, **kwargs):
            called.append(True)
            raise AssertionError("Should not have called requests.post")

        monkeypatch.setattr(licensing.requests, "post", fake_post)
        result = licensing._attempt_heartbeat_refresh()
        assert result is False
        assert called == []

    def test_refresh_network_error_returns_false(self, configured_module, monkeypatch):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key))

        def fake_post(*args, **kwargs):
            raise licensing.requests.ConnectionError("network down")

        monkeypatch.setattr(licensing.requests, "post", fake_post)
        result = licensing._attempt_heartbeat_refresh()
        assert result is False

    def test_refresh_revoked_license_returns_false(self, configured_module, monkeypatch):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key))

        class FakeResponse:
            status_code = 403
            content = b'{"error": "License has been revoked"}'
            text = '{"error": "License has been revoked"}'

        monkeypatch.setattr(licensing.requests, "post", lambda *a, **kw: FakeResponse())
        result = licensing._attempt_heartbeat_refresh()
        assert result is False

    def test_refresh_bad_signature_does_not_save(self, configured_module, monkeypatch):
        private_key, _ = configured_module
        licensing.save_license(_sign_license(private_key))

        # Hub returns a token signed with the WRONG key
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        bad_key = Ed25519PrivateKey.generate()
        bad_heartbeat = _sign_heartbeat(bad_key)

        class FakeResponse:
            status_code = 200
            content = bad_heartbeat
            text = bad_heartbeat.decode()

        monkeypatch.setattr(licensing.requests, "post", lambda *a, **kw: FakeResponse())
        result = licensing._attempt_heartbeat_refresh()
        assert result is False
        # Heartbeat file should NOT exist (we haven't installed one)
        assert not licensing.HEARTBEAT_PATH.exists()
