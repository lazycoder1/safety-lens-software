"""Canonical redaction for API responses, audit rows, and diagnostics."""

from __future__ import annotations

import re
from copy import deepcopy
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED_VALUE = "***redacted***"

_SENSITIVE_KEYS = {
    "accesstoken",
    "apikey",
    "authorization",
    "bottoken",
    "clientsecret",
    "cookie",
    "jwtsecret",
    "password",
    "passwd",
    "privatekey",
    "proxyauthorization",
    "refreshtoken",
    "secret",
    "setcookie",
    "smtppass",
}
_SENSITIVE_QUERY_KEYS = _SENSITIVE_KEYS | {"key", "sig", "signature", "token"}
_HEADER_CONTAINER_KEYS = {"header", "headers", "httpheaders"}
_NON_SECRET_TOKEN_KEYS = {"maxtoken", "maxtokens"}
_URL_SECRET_QUERY_PATTERN = re.compile(
    r"([?&](?:access_token|api_key|apikey|key|password|secret|sig|signature|token)=)[^&#\s\"'<>]+",
    flags=re.I,
)


def _normalize_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def is_redacted(value: object) -> bool:
    return isinstance(value, str) and value == REDACTED_VALUE


def _is_sensitive_key(normalized_key: str) -> bool:
    if normalized_key in _SENSITIVE_KEYS:
        return True
    if normalized_key in _NON_SECRET_TOKEN_KEYS:
        return False
    return normalized_key.endswith(
        ("password", "passwd", "secret", "token", "apikey", "privatekey")
    )


def redact_text_secrets(text: str, secret_values=()) -> str:
    """Redact known secret values and common credential-bearing URL shapes."""
    redacted = text
    known_values: set[str] = set()
    for value in secret_values:
        if value in (None, ""):
            continue
        secret = str(value)
        known_values.add(secret)
        try:
            parsed_secret = urlsplit(secret)
        except ValueError:
            continue
        if parsed_secret.scheme and parsed_secret.netloc and parsed_secret.path not in ("", "/"):
            known_values.add(parsed_secret.path)
            if parsed_secret.query:
                known_values.add(f"{parsed_secret.path}?{parsed_secret.query}")
    for secret in sorted(
        known_values,
        key=len,
        reverse=True,
    ):
        redacted = redacted.replace(secret, REDACTED_VALUE)
    redacted = re.sub(
        r"([a-z][a-z0-9+.-]*://)[^/@\s]+@",
        rf"\1{REDACTED_VALUE}@",
        redacted,
        flags=re.I,
    )
    redacted = re.sub(
        r"(https://api\.telegram\.org/bot)[^/\s\"'<>]+",
        rf"\1{REDACTED_VALUE}",
        redacted,
        flags=re.I,
    )
    return _URL_SECRET_QUERY_PATTERN.sub(
        lambda match: f"{match.group(1)}{REDACTED_VALUE}",
        redacted,
    )


def _redact_url_credentials(value: str) -> str:
    if "://" not in value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return redact_text_secrets(value)
    if not parsed.scheme or not parsed.netloc:
        return value

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    try:
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
    except ValueError:
        return redact_text_secrets(value)

    query = []
    for query_key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        if _normalize_key(query_key) in _SENSITIVE_QUERY_KEYS and query_value:
            query_value = REDACTED_VALUE
        query.append((query_key, query_value))
    return redact_text_secrets(
        urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query), parsed.fragment))
    )


def redact_sensitive_data(value, *, _key: object | None = None):
    """Return a deep redacted copy without mutating the caller's object."""
    normalized_key = _normalize_key(_key) if _key is not None else ""
    if _is_sensitive_key(normalized_key):
        return REDACTED_VALUE if value not in (None, "", [], {}) else deepcopy(value)

    if isinstance(value, dict):
        if normalized_key in _HEADER_CONTAINER_KEYS:
            return {
                str(header): REDACTED_VALUE if header_value not in (None, "", [], {}) else deepcopy(header_value)
                for header, header_value in value.items()
            }
        return {
            key: redact_sensitive_data(item, _key=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    if isinstance(value, str):
        return _redact_url_credentials(value)
    return deepcopy(value)
