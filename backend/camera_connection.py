"""Helpers for structured RTSP camera connection data and redaction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
from urllib.parse import quote, unquote, urlparse

DEFAULT_RTSP_PORT = 554
DEFAULT_ONVIF_PORT = 80


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def parse_rtsp_url(rtsp_url: str) -> dict[str, Any]:
    raw_url = _clean_string(rtsp_url)
    if not raw_url:
        return {}
    if "://" not in raw_url:
        raw_url = f"rtsp://{raw_url}"
    parsed = urlparse(raw_url)
    stream_path = parsed.path or ""
    if parsed.query:
        stream_path = f"{stream_path}?{parsed.query}"
    return {
        "scheme": parsed.scheme or "rtsp",
        "host": parsed.hostname or "",
        "rtsp_port": parsed.port or DEFAULT_RTSP_PORT,
        "stream_path": stream_path or "",
        "username": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
    }


def normalize_stream_path(stream_path: str) -> str:
    cleaned = _clean_string(stream_path)
    if not cleaned:
        return ""
    if cleaned.startswith("?"):
        return f"/{cleaned}"
    if cleaned.startswith("/"):
        return cleaned
    return f"/{cleaned}"


def infer_preferred_stream(stream_path: str) -> str:
    normalized = normalize_stream_path(stream_path).lower()
    if not normalized:
        return "main"
    if any(marker in normalized for marker in ("sub", "102", "stream2", "subtype=1", "subtype=01")):
        return "sub"
    if "mjpeg" in normalized or normalized.endswith(".mjpg"):
        return "mjpeg"
    return "main"


def credentials_configured(camera: Mapping[str, Any]) -> bool:
    return bool(_clean_string(camera.get("username")) or _clean_string(camera.get("password")))


def build_rtsp_url(camera: Mapping[str, Any], *, include_credentials: bool = True) -> str:
    host = _clean_string(camera.get("host"))
    if host:
        scheme = _clean_string(camera.get("scheme")) or "rtsp"
        rtsp_port = camera.get("rtsp_port") or DEFAULT_RTSP_PORT
        stream_path = normalize_stream_path(camera.get("stream_path", ""))
        auth_prefix = ""
        username = _clean_string(camera.get("username"))
        password = _clean_string(camera.get("password"))
        if include_credentials and username:
            auth_prefix = quote(username, safe="")
            if password:
                auth_prefix = f"{auth_prefix}:{quote(password, safe='')}"
            auth_prefix = f"{auth_prefix}@"
        return f"{scheme}://{auth_prefix}{host}:{rtsp_port}{stream_path}"

    rtsp_url = _clean_string(camera.get("rtsp_url"))
    if include_credentials or not rtsp_url:
        return rtsp_url
    parsed = parse_rtsp_url(rtsp_url)
    if not parsed:
        return rtsp_url
    return build_rtsp_url(parsed, include_credentials=False)


def connection_summary(camera: Mapping[str, Any]) -> str:
    return build_rtsp_url(camera, include_credentials=False)


def public_camera_connection_fields(camera: Mapping[str, Any]) -> dict[str, Any]:
    host = _clean_string(camera.get("host"))
    rtsp_port = camera.get("rtsp_port")
    stream_path = normalize_stream_path(camera.get("stream_path", ""))
    safe_rtsp_url = build_rtsp_url(camera, include_credentials=False)
    if not host and safe_rtsp_url:
        parsed = parse_rtsp_url(safe_rtsp_url)
        host = parsed.get("host", "")
        rtsp_port = parsed.get("rtsp_port")
        stream_path = normalize_stream_path(parsed.get("stream_path", ""))
    return {
        "host": host,
        "rtsp_port": rtsp_port or (DEFAULT_RTSP_PORT if host else None),
        "onvif_port": camera.get("onvif_port"),
        "stream_path": stream_path,
        "preferred_stream": _clean_string(camera.get("preferred_stream")) or infer_preferred_stream(stream_path),
        "onvif_uuid": _clean_string(camera.get("onvif_uuid")) or None,
        "discovery_fingerprint": _clean_string(camera.get("discovery_fingerprint")) or None,
        "credentials_configured": credentials_configured(camera),
        "connection_summary": safe_rtsp_url or None,
        "rtsp_url": safe_rtsp_url,
    }


def redact_camera_secrets(camera: Mapping[str, Any]) -> dict[str, Any]:
    redacted = deepcopy(dict(camera))
    redacted.pop("username", None)
    redacted.pop("password", None)
    redacted.update(public_camera_connection_fields(camera))
    return redacted


def normalize_camera_connection(camera: dict[str, Any], *, existing_camera: Mapping[str, Any] | None = None) -> bool:
    changed = False
    if camera.get("stream_type", "file") != "rtsp":
        return changed

    parsed = parse_rtsp_url(camera.get("rtsp_url", ""))
    for field in ("host", "rtsp_port", "stream_path", "username", "password"):
        if field in parsed and not _clean_string(camera.get(field)) and parsed[field]:
            camera[field] = parsed[field]
            changed = True

    if existing_camera:
        for secret_field in ("username", "password"):
            if secret_field not in camera or camera.get(secret_field) is None:
                existing_value = _clean_string(existing_camera.get(secret_field))
                if existing_value:
                    camera[secret_field] = existing_value
                    changed = True

    if _clean_string(camera.get("stream_path")):
        normalized_path = normalize_stream_path(camera.get("stream_path", ""))
        if camera.get("stream_path") != normalized_path:
            camera["stream_path"] = normalized_path
            changed = True

    if _clean_string(camera.get("host")) and not camera.get("rtsp_port"):
        camera["rtsp_port"] = DEFAULT_RTSP_PORT
        changed = True

    if _clean_string(camera.get("host")) and not camera.get("onvif_port") and _clean_string(camera.get("onvif_uuid")):
        camera["onvif_port"] = DEFAULT_ONVIF_PORT
        changed = True

    preferred_stream = _clean_string(camera.get("preferred_stream"))
    inferred = infer_preferred_stream(camera.get("stream_path", ""))
    if preferred_stream != inferred and not preferred_stream:
        camera["preferred_stream"] = inferred
        changed = True

    sanitized_rtsp_url = build_rtsp_url(camera, include_credentials=False)
    if sanitized_rtsp_url and camera.get("rtsp_url") != sanitized_rtsp_url:
        camera["rtsp_url"] = sanitized_rtsp_url
        changed = True

    return changed
