"""Tests for persistent runtime secrets and canonical redaction."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from zipfile import ZipFile

import jwt
import pytest

import auth_store
import audit_store
import config_manager
import diagnostics
from secret_redaction import REDACTED_VALUE, redact_sensitive_data, redact_text_secrets
from scripts.migrate_config_to_postgres import write_backup


def test_generated_jwt_secret_survives_process_restart(tmp_path, monkeypatch):
    secret_path = tmp_path / "runtime" / "auth" / "jwt.secret"
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_FILE", raising=False)
    monkeypatch.setattr(auth_store, "JWT_SECRET_PATH", secret_path)

    first = auth_store._load_jwt_secret()
    token = jwt.encode({"sub": "restart-test"}, first, algorithm=auth_store.JWT_ALGORITHM)
    second = auth_store._load_jwt_secret()

    assert first == second
    assert jwt.decode(token, second, algorithms=[auth_store.JWT_ALGORITHM])["sub"] == "restart-test"
    assert secret_path.stat().st_mode & 0o777 == 0o600
    assert secret_path.parent.stat().st_mode & 0o777 == 0o700


def test_legacy_config_jwt_secret_migrates_without_invalidating_tokens(tmp_path, monkeypatch):
    secret_path = tmp_path / "runtime" / "auth" / "jwt.secret"
    legacy = "legacy-secret-that-is-long-enough-1234567890"
    config = {"auth": {"jwt_secret": legacy}, "cameras": {}}
    removed = []
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_FILE", raising=False)
    monkeypatch.setattr(auth_store, "JWT_SECRET_PATH", secret_path)
    monkeypatch.setattr(auth_store, "JWT_SECRET", None)
    monkeypatch.setattr(config_manager, "get_config", lambda: config)

    def remove_legacy(expected):
        removed.append(expected)
        config.pop("auth", None)
        return True

    monkeypatch.setattr(config_manager, "remove_legacy_jwt_secret", remove_legacy)

    token = jwt.encode({"sub": "upgrade-test"}, legacy, algorithm=auth_store.JWT_ALGORITHM)
    migrated = auth_store.init_jwt_secret()

    assert migrated == legacy
    assert secret_path.read_text().strip() == legacy
    assert jwt.decode(token, migrated, algorithms=[auth_store.JWT_ALGORITHM])["sub"] == "upgrade-test"
    assert "auth" not in config
    assert removed == [legacy]


def test_mismatched_runtime_and_legacy_jwt_secrets_fail_closed(tmp_path, monkeypatch):
    secret_path = tmp_path / "runtime" / "auth" / "jwt.secret"
    secret_path.parent.mkdir(parents=True)
    secret_path.write_text("runtime-secret-that-is-long-enough-123456\n")
    legacy = "different-legacy-secret-that-is-long-enough"
    config = {"auth": {"jwt_secret": legacy}}
    removed = []
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_FILE", raising=False)
    monkeypatch.setattr(auth_store, "JWT_SECRET_PATH", secret_path)
    monkeypatch.setattr(auth_store, "JWT_SECRET", None)
    monkeypatch.setattr(config_manager, "get_config", lambda: config)
    monkeypatch.setattr(config_manager, "remove_legacy_jwt_secret", lambda value: removed.append(value))

    with pytest.raises(RuntimeError, match="does not match legacy"):
        auth_store.init_jwt_secret()

    assert config["auth"]["jwt_secret"] == legacy
    assert removed == []


def test_legacy_jwt_secret_with_surrounding_whitespace_fails_before_migration(tmp_path, monkeypatch):
    secret_path = tmp_path / "runtime" / "auth" / "jwt.secret"
    legacy = "  legacy-secret-that-is-long-enough-123456  "
    config = {"auth": {"jwt_secret": legacy}}
    removed = []
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_FILE", raising=False)
    monkeypatch.setattr(auth_store, "JWT_SECRET_PATH", secret_path)
    monkeypatch.setattr(auth_store, "JWT_SECRET", None)
    monkeypatch.setattr(config_manager, "get_config", lambda: config)
    monkeypatch.setattr(config_manager, "remove_legacy_jwt_secret", lambda value: removed.append(value))

    with pytest.raises(RuntimeError, match="surrounding whitespace"):
        auth_store.init_jwt_secret()

    assert not secret_path.exists()
    assert removed == []


def test_concurrent_jwt_initializers_share_one_persisted_secret(tmp_path, monkeypatch):
    secret_path = tmp_path / "runtime" / "auth" / "jwt.secret"
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_FILE", raising=False)
    monkeypatch.setattr(auth_store, "JWT_SECRET_PATH", secret_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: auth_store._load_jwt_secret(), range(32)))

    assert len(set(results)) == 1
    assert auth_store._read_jwt_secret_file(secret_path, str(secret_path)) == results[0]


def test_explicit_missing_jwt_secret_file_fails_startup(tmp_path, monkeypatch):
    secret_path = tmp_path / "missing.secret"
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("JWT_SECRET_FILE", str(secret_path))
    monkeypatch.setattr(auth_store, "JWT_SECRET_PATH", secret_path)

    with pytest.raises(RuntimeError, match="does not exist"):
        auth_store._load_jwt_secret()


def test_weak_explicit_jwt_secret_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "too-short")

    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        auth_store._load_jwt_secret()


def test_explicit_jwt_secret_with_surrounding_whitespace_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "  explicit-secret-that-is-long-enough-123456  ")

    with pytest.raises(RuntimeError, match="surrounding whitespace"):
        auth_store._load_jwt_secret()


def test_recursive_redaction_covers_headers_credentials_and_url_tokens():
    raw = {
        "password": "camera-password-sentinel",
        "headers": {
            "Authorization": "Bearer authorization-sentinel",
            "X-Custom-Key": "custom-header-sentinel",
        },
        "video": "rtsp://viewer:rtsp-secret@192.0.2.10/live?token=query-secret&profile=sub",
        "provider_access_token": "provider-token-sentinel",
        "max_tokens": 300,
    }

    redacted = redact_sensitive_data(raw)
    serialized = json.dumps(redacted)

    for sentinel in (
        "camera-password-sentinel",
        "authorization-sentinel",
        "custom-header-sentinel",
        "rtsp-secret",
        "query-secret",
        "provider-token-sentinel",
    ):
        assert sentinel not in serialized
    assert set(redacted["headers"].values()) == {REDACTED_VALUE}
    assert redacted["max_tokens"] == 300
    assert raw["password"] == "camera-password-sentinel"


def test_malformed_urls_fail_closed_during_redaction():
    malformed = "rtsp://viewer:password-sentinel@example.test:bad/live?token=query-sentinel"

    redacted = redact_sensitive_data({"video": malformed})["video"]

    assert "password-sentinel" not in redacted
    assert "query-sentinel" not in redacted
    assert REDACTED_VALUE in redacted


def test_text_redaction_removes_provider_path_secrets():
    telegram = "https://api.telegram.org/bottelegram-path-sentinel/sendMessage"
    webhook = "https://hooks.example/services/webhook-path-sentinel"

    redacted = redact_text_secrets(f"{telegram} {webhook}", [webhook])

    assert "telegram-path-sentinel" not in redacted
    assert "webhook-path-sentinel" not in redacted


def test_text_redaction_removes_requests_style_relative_webhook_path():
    webhook = "https://hooks.example/services/account/webhook-path-sentinel?key=query-sentinel"
    error = (
        "HTTPSConnectionPool(host='hooks.example'): failed with "
        "url: /services/account/webhook-path-sentinel?key=query-sentinel"
    )

    redacted = redact_text_secrets(error, [webhook])

    assert "webhook-path-sentinel" not in redacted
    assert "query-sentinel" not in redacted


def test_diagnostics_bundle_is_private_and_scrubs_log_secrets(tmp_path, monkeypatch):
    diagnostics_dir = tmp_path / "diagnostics"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    secret = "diagnostics-webhook-path-sentinel"
    (logs_dir / "safetylens.log").write_text(
        f"request failed with url: /services/{secret}\n"
    )
    raw_config = {
        "database": {},
        "telegram": {},
        "email": {},
        "webhook": {"url": f"https://hooks.example/services/{secret}", "headers": {}},
        "cameras": {},
    }
    monkeypatch.setattr(diagnostics, "DIAGNOSTICS_DIR", diagnostics_dir)
    monkeypatch.setattr(diagnostics, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(diagnostics, "get_config", lambda: raw_config)
    monkeypatch.setattr(diagnostics, "get_redacted_config", lambda: {"webhook": {"url": REDACTED_VALUE}})
    monkeypatch.setattr(diagnostics, "build_health_snapshot", lambda: {"status": "ok"})
    monkeypatch.setattr(diagnostics.alert_store, "get_alerts", lambda limit: [])
    monkeypatch.setattr(diagnostics.alert_store, "get_stats", lambda: {})
    monkeypatch.setattr(diagnostics.audit_store, "get_recent", lambda limit: [])
    monkeypatch.setattr(
        diagnostics.licensing,
        "get_status",
        lambda: SimpleNamespace(to_public_dict=lambda: {"state": "valid"}),
    )

    bundle = diagnostics.create_diagnostics_bundle()

    assert diagnostics_dir.stat().st_mode & 0o777 == 0o700
    assert bundle.stat().st_mode & 0o777 == 0o600
    with ZipFile(bundle) as archive:
        log_text = archive.read("logs/safetylens.log").decode()
    assert secret not in log_text
    assert REDACTED_VALUE in log_text


def test_postgres_migration_backup_is_private(tmp_path):
    backup = tmp_path / ".deploy" / "app_config.backup.json"

    write_backup(backup, {"telegram": {"bot_token": "backup-secret"}})

    assert json.loads(backup.read_text())["telegram"]["bot_token"] == "backup-secret"
    assert backup.stat().st_mode & 0o777 == 0o600
    assert backup.parent.stat().st_mode & 0o777 == 0o700


def test_audit_store_redacts_before_database_persistence(monkeypatch):
    captured = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _query, params):
            captured["details"] = params[-1]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            return None

    monkeypatch.setattr(audit_store, "get_conn", FakeConnection)

    result = audit_store.log_event(
        "camera.update",
        details={
            "updates": {
                "password": "audit-password-sentinel",
                "headers": {"X-Key": "audit-header-sentinel"},
            }
        },
    )

    serialized = captured["details"] + json.dumps(result)
    assert "audit-password-sentinel" not in serialized
    assert "audit-header-sentinel" not in serialized
    assert REDACTED_VALUE in serialized
