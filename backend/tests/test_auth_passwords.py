import asyncio
import json

import pytest
from fastapi import HTTPException

import auth_store
from routers.admin import ResetPasswordRequest, api_reset_password
from routers.auth import (
    ChangePasswordRequest,
    RegisterRequest,
    api_change_password,
    api_register,
)


def _ascii_password(byte_length: int) -> str:
    prefix = "Aa1!"
    return prefix + ("x" * (byte_length - len(prefix)))


def test_validate_password_accepts_exact_bcrypt_byte_limit():
    password = _ascii_password(auth_store.BCRYPT_MAX_PASSWORD_BYTES)

    assert len(password.encode("utf-8")) == 72
    assert auth_store.validate_password(password) == (True, None)

    password_hash = auth_store._hash_password(password)
    assert auth_store._verify_password(password, password_hash) is True


def test_validate_password_rejects_password_over_bcrypt_byte_limit():
    password = _ascii_password(auth_store.BCRYPT_MAX_PASSWORD_BYTES + 1)

    valid, error = auth_store.validate_password(password)

    assert valid is False
    assert error == "Password must be at most 72 bytes when encoded as UTF-8"
    with pytest.raises(ValueError, match="at most 72 bytes"):
        auth_store._hash_password(password)


def test_validate_password_counts_utf8_bytes_not_characters():
    exact_limit = "Aa1!" + ("é" * 34)
    over_limit = exact_limit + "é"

    assert len(exact_limit.encode("utf-8")) == 72
    assert auth_store.validate_password(exact_limit) == (True, None)
    assert len(over_limit) < auth_store.BCRYPT_MAX_PASSWORD_BYTES
    assert auth_store.validate_password(over_limit) == (
        False,
        "Password must be at most 72 bytes when encoded as UTF-8",
    )


def test_verify_password_preserves_legacy_raw_byte_truncation():
    password = "Aa1!" + ("x" * 67) + "é"
    encoded = password.encode("utf-8")
    assert len(encoded) == 73

    # bcrypt before 5.0 silently hashed only the first 72 raw bytes, even if
    # that split a UTF-8 sequence. Existing users must still be able to verify
    # the historical hash so authenticate() can require a password rotation.
    legacy_hash = auth_store.bcrypt.hashpw(
        encoded[: auth_store.BCRYPT_MAX_PASSWORD_BYTES],
        auth_store.bcrypt.gensalt(),
    ).decode()

    assert auth_store._verify_password(password, legacy_hash) is True


def test_verify_password_returns_false_for_invalid_stored_hash():
    assert auth_store._verify_password("Aa1!password", "not-a-bcrypt-hash") is False


def test_invalid_unicode_is_rejected_without_encoding_exception():
    password = "\ud800Aa1!aaaa"

    assert auth_store.validate_password(password) == (
        False,
        "Password contains invalid Unicode characters",
    )
    with pytest.raises(ValueError, match="invalid Unicode"):
        auth_store._hash_password(password)
    assert auth_store._verify_password(password, "not-used") is False


def test_legacy_oversized_login_requires_password_rotation(monkeypatch):
    password = "Aa1!" + ("x" * 67) + "é"
    encoded = password.encode("utf-8")
    legacy_hash = auth_store.bcrypt.hashpw(
        encoded[: auth_store.BCRYPT_MAX_PASSWORD_BYTES],
        auth_store.bcrypt.gensalt(),
    ).decode()
    row = {
        "id": "legacy-user",
        "username": "legacy",
        "password_hash": legacy_hash,
        "role": "operator",
        "status": "active",
        "must_change_password": False,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_login": None,
    }
    statements = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params):
            statements.append((query, params))

        def fetchone(self):
            return row

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self, **_kwargs):
            return FakeCursor()

        def commit(self):
            return None

    monkeypatch.setattr(auth_store, "get_conn", FakeConnection)

    user = auth_store.authenticate("legacy", password)

    assert user is not None
    assert user["mustChangePassword"] is True
    update_query, update_params = statements[-1]
    assert "must_change_password = CASE WHEN %s" in update_query
    assert update_params[1:] == (True, "legacy-user")


def test_register_returns_400_for_oversized_password(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("create_user must not receive an oversized password")

    monkeypatch.setattr(auth_store, "create_user", fail_if_called)

    response = asyncio.run(
        api_register(RegisterRequest(username="operator", password=_ascii_password(73)))
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "detail": "Password must be at most 72 bytes when encoded as UTF-8"
    }


def test_register_returns_400_for_invalid_unicode(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("create_user must not receive invalid Unicode")

    monkeypatch.setattr(auth_store, "create_user", fail_if_called)

    response = asyncio.run(
        api_register(RegisterRequest(username="operator", password="\ud800Aa1!aaaa"))
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "detail": "Password contains invalid Unicode characters"
    }


def test_change_password_returns_400_before_mutating_oversized_password(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("change_password must not receive an oversized new password")

    monkeypatch.setattr(auth_store, "change_password", fail_if_called)

    response = asyncio.run(
        api_change_password(
            None,
            ChangePasswordRequest(
                currentPassword="current",
                newPassword=_ascii_password(73),
            ),
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "detail": "Password must be at most 72 bytes when encoded as UTF-8"
    }


def test_admin_reset_returns_400_before_mutating_oversized_password(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("reset_password must not receive an oversized new password")

    monkeypatch.setattr(auth_store, "reset_password", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            api_reset_password(
                "user-1",
                ResetPasswordRequest(newPassword=_ascii_password(73)),
                None,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Password must be at most 72 bytes when encoded as UTF-8"
