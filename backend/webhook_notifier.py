"""
Webhook alert notifications for SafetyLens.
HTTP POST JSON payloads to a configurable endpoint.

Synchronous provider call, fenced and retried by the durable outbox worker.
"""

import base64
import hashlib
import http.client
import ipaddress
import json as jsonlib
import logging
import os
import socket
import ssl
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

from config_manager import get_config
from delivery_result import (
    DeliveryDisposition,
    ProviderDeliveryResult,
    parse_retry_after,
    stable_delivery_identity,
)
from secret_redaction import REDACTED_VALUE

logger = logging.getLogger("safetylens.webhook")

_FORBIDDEN_REQUEST_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class _WebhookPolicyError(ValueError):
    """A sanitized, operator-actionable webhook policy rejection."""

    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True)
class _ValidatedWebhookTarget:
    """One DNS-validated endpoint used unchanged through the HTTP send."""

    url: str
    scheme: str
    hostname: str
    port: int
    address: ipaddress.IPv4Address | ipaddress.IPv6Address
    request_target: str
    host_header: str


class _PinnedResponse:
    """Small response facade that owns its HTTP connection."""

    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPConnection,
    ) -> None:
        self.status_code = response.status
        self.headers = response.headers
        self._response = response
        self._connection = connection

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()


def send_alert(alert: dict, snapshot_path: str | None = None) -> bool:
    """POST an alert and return whether the endpoint accepted it."""
    return send_alert_result(alert, snapshot_path).success


def send_alert_result(
    alert: dict,
    snapshot_path: str | None = None,
    *,
    include_snapshot_override: bool | None = None,
    webhook_config_override: dict | None = None,
    url_override: str | None = None,
) -> ProviderDeliveryResult:
    """POST an alert and distinguish retryable transport failures from rejections."""
    url = ""
    try:
        wh = (
            deepcopy(webhook_config_override)
            if webhook_config_override is not None
            else deepcopy(get_config().get("webhook", {}))
        )

        if not wh.get("enabled", False):
            return ProviderDeliveryResult(
                DeliveryDisposition.SKIPPED,
                "Webhook is disabled",
                error_code="channel_disabled",
            )

        url = url_override if url_override is not None else wh.get("url", "")
        if not url:
            return ProviderDeliveryResult(
                DeliveryDisposition.TERMINAL,
                "Webhook configuration is incomplete",
                error_code="invalid_configuration",
            )

        severity_filter = wh.get("severities", ["P1", "P2"])
        if alert.get("severity") not in severity_filter:
            return ProviderDeliveryResult(
                DeliveryDisposition.SKIPPED,
                "Alert severity is filtered",
                error_code="severity_filtered",
            )

        target = _validate_webhook_url(url)

        include_snapshot = (
            bool(wh.get("include_snapshot", False))
            if include_snapshot_override is None
            else bool(include_snapshot_override)
        )
        payload = _build_payload(
            alert,
            snapshot_path,
            include_snapshot=include_snapshot,
        )
        headers = _validated_headers(wh.get("headers", {}))
        headers["Content-Type"] = "application/json"
        headers["Idempotency-Key"] = _idempotency_key(alert, url)

        response = _post_pinned(
            url,
            json=payload,
            headers=headers,
            timeout=10,
            allow_redirects=False,
            stream=True,
            verify=True,
            target=target,
        )
        try:
            result = _classify_response(response)
        finally:
            _close_response(response)
        if not result.success:
            logger.warning(
                "Webhook rejected alert",
                extra={
                    "alert_id": alert.get("id"),
                    "url": REDACTED_VALUE,
                    "error_code": result.error_code,
                    "provider_status": result.provider_status,
                    "retry_after_seconds": result.retry_after_seconds,
                },
            )
            return result

        logger.info("Webhook alert sent", extra={"alert_id": alert.get("id"), "url": REDACTED_VALUE})
        return result
    except _WebhookPolicyError as exc:
        logger.error(
            "Webhook endpoint rejected by outbound request policy",
            extra={"error_code": exc.error_code},
        )
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "Webhook endpoint or headers violate outbound request policy",
            error_code=exc.error_code,
        )
    except socket.gaierror:
        logger.warning("Webhook hostname lookup failed")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Webhook hostname could not be resolved temporarily",
            error_code="dns_resolution_failed",
        )
    except (
        requests.exceptions.InvalidSchema,
        requests.exceptions.InvalidURL,
        requests.exceptions.InvalidHeader,
        requests.exceptions.MissingSchema,
        requests.exceptions.TooManyRedirects,
    ):
        logger.error("Webhook endpoint configuration is invalid")
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "Webhook endpoint configuration is invalid",
            error_code="invalid_endpoint",
        )
    except requests.exceptions.SSLError:
        logger.warning("Webhook TLS transport failed")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Webhook TLS transport failed temporarily",
            error_code="tls_transport_error",
            acceptance_unknown=True,
        )
    except requests.exceptions.ConnectTimeout:
        logger.warning("Webhook connection timed out")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Webhook is temporarily unreachable",
            error_code="connect_timeout",
        )
    except requests.exceptions.ReadTimeout:
        logger.warning("Webhook response timed out")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Webhook acceptance could not be confirmed",
            error_code="read_timeout",
            acceptance_unknown=True,
        )
    except requests.exceptions.ConnectionError:
        logger.warning("Webhook connection failed")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Webhook acceptance could not be confirmed",
            error_code="connection_error",
            acceptance_unknown=True,
        )
    except requests.exceptions.RequestException:
        logger.warning("Webhook request failed")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Webhook request failed temporarily",
            error_code="request_error",
            acceptance_unknown=True,
        )
    except Exception:
        logger.error("Webhook notification failed unexpectedly")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Webhook delivery failed unexpectedly",
            error_code="unexpected_error",
            acceptance_unknown=True,
        )


def test_connection(url: str, headers: dict | None = None) -> dict:
    """Send a test payload to the webhook URL."""
    try:
        test_payload = {
            "type": "test",
            "source": "SafetyLens",
            "message": "Webhook connection test successful.",
        }
        target = _validate_webhook_url(url)
        req_headers = _validated_headers(headers or {})
        req_headers["Content-Type"] = "application/json"

        resp = _post_pinned(
            url,
            json=test_payload,
            headers=req_headers,
            timeout=10,
            allow_redirects=False,
            stream=True,
            verify=True,
            target=target,
        )
        try:
            if 200 <= resp.status_code < 300:
                return {"ok": True}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        finally:
            _close_response(resp)
    except Exception:
        return {"ok": False, "error": "Webhook connection test failed"}


def _classify_response(response: object) -> ProviderDeliveryResult:
    status = int(response.status_code)
    if 200 <= status < 300:
        return ProviderDeliveryResult(
            DeliveryDisposition.DELIVERED,
            "Delivered",
            provider_status=status,
        )
    retryable = status in {408, 425, 429} or 500 <= status < 600
    headers = getattr(response, "headers", {})
    retry_after = None
    if retryable and hasattr(headers, "get"):
        retry_after = parse_retry_after(headers.get("Retry-After"))
    return ProviderDeliveryResult(
        DeliveryDisposition.RETRYABLE if retryable else DeliveryDisposition.TERMINAL,
        f"Webhook returned HTTP {status}",
        error_code=f"http_{status}",
        provider_status=status,
        retry_after_seconds=retry_after,
        acceptance_unknown=(status == 408 or 500 <= status < 600),
    )


def _idempotency_key(alert: dict, url: str) -> str:
    target_fingerprint = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"safetylens-{stable_delivery_identity(alert)}-{target_fingerprint}"


def _build_payload(alert: dict, snapshot_path: str | None, include_snapshot: bool) -> dict:
    """Build the JSON payload to POST."""
    payload = {
        "source": "SafetyLens",
        "event": "alert",
        "alert": {
            "id": alert.get("id"),
            "severity": alert.get("severity"),
            "rule": alert.get("rule"),
            "cameraId": alert.get("cameraId"),
            "cameraName": alert.get("cameraName"),
            "zone": alert.get("zone"),
            "confidence": alert.get("confidence"),
            "timestamp": alert.get("timestamp"),
            "description": alert.get("description"),
        },
    }

    if include_snapshot and snapshot_path:
        try:
            data = Path(snapshot_path).read_bytes()
            payload["snapshot_base64"] = base64.b64encode(data).decode("ascii")
        except FileNotFoundError:
            pass

    return payload


def _allowed_hosts(env_name: str) -> set[str]:
    return {
        _normalize_hostname(value)
        for value in os.getenv(env_name, "").split(",")
        if value.strip()
    }


def _normalize_hostname(hostname: str) -> str:
    return str(hostname).strip().lower().rstrip(".")


def _validate_webhook_url(url: str) -> _ValidatedWebhookTarget:
    """Resolve and reject targets that could reach local or privileged services."""
    if not isinstance(url, str) or not url:
        raise _WebhookPolicyError("invalid_endpoint")
    if "\\" in url or any(ord(character) < 32 for character in url):
        raise _WebhookPolicyError("invalid_endpoint")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError, UnicodeError):
        raise _WebhookPolicyError("invalid_endpoint") from None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not hostname:
        raise _WebhookPolicyError("invalid_endpoint")
    if parsed.username is not None or parsed.password is not None:
        raise _WebhookPolicyError("endpoint_userinfo_forbidden")
    if parsed.fragment:
        raise _WebhookPolicyError("endpoint_fragment_forbidden")

    normalized_host = _normalize_hostname(hostname)
    if not normalized_host:
        raise _WebhookPolicyError("invalid_endpoint")
    if "%" in normalized_host:
        # Scoped IPv6 addresses are local-interface concepts and must never be
        # accepted as an Internet webhook destination.
        raise _WebhookPolicyError("invalid_endpoint")
    try:
        ascii_hostname = normalized_host.encode("idna").decode("ascii")
    except UnicodeError:
        raise _WebhookPolicyError("invalid_endpoint") from None
    if scheme != "https" and normalized_host not in _allowed_hosts(
        "WEBHOOK_ALLOWED_HTTP_HOSTS"
    ):
        raise _WebhookPolicyError("endpoint_https_required")

    if port is None:
        port = 443 if scheme == "https" else 80
    if not 1 <= port <= 65535:
        raise _WebhookPolicyError("invalid_endpoint")

    addresses = _resolve_addresses(ascii_hostname, port)
    private_allowed = normalized_host in _allowed_hosts(
        "WEBHOOK_ALLOWED_PRIVATE_HOSTS"
    )
    if not private_allowed and any(_is_non_public(address) for address in addresses):
        raise _WebhookPolicyError("endpoint_address_forbidden")

    path = quote(
        parsed.path or "/",
        safe="/%:@!$&'()*+,;=-._~",
    )
    query = quote(
        parsed.query,
        safe="=&?/:;+,%@!$'()*-._~",
    )
    request_target = f"{path}?{query}" if query else path
    default_port = 443 if scheme == "https" else 80
    try:
        hostname_is_ipv6 = ipaddress.ip_address(ascii_hostname).version == 6
    except ValueError:
        hostname_is_ipv6 = False
    authority_host = f"[{ascii_hostname}]" if hostname_is_ipv6 else ascii_hostname
    host_header = (
        authority_host
        if port == default_port
        else f"{authority_host}:{port}"
    )
    return _ValidatedWebhookTarget(
        url=url,
        scheme=scheme,
        hostname=ascii_hostname,
        port=port,
        address=addresses[0],
        request_target=request_target,
        host_header=host_header,
    )


def _post_pinned(
    url: str,
    *,
    json: dict,
    headers: dict[str, str],
    timeout: float,
    allow_redirects: bool,
    stream: bool,
    verify: bool,
    target: _ValidatedWebhookTarget,
) -> _PinnedResponse:
    """POST over a socket pinned to the address approved by URL validation.

    A raw socket is intentional: high-level clients resolve the hostname a
    second time and may also honor process proxy variables. The HTTP Host
    header and TLS SNI remain the configured hostname, so normal certificate
    verification is preserved while DNS rebinding is eliminated.
    """
    if (
        url != target.url
        or allow_redirects
        or not stream
        or not verify
    ):
        raise _WebhookPolicyError("invalid_transport_options")

    try:
        total_timeout = float(timeout)
    except (TypeError, ValueError):
        raise _WebhookPolicyError("invalid_transport_options") from None
    if total_timeout <= 0:
        raise _WebhookPolicyError("invalid_transport_options")

    body = jsonlib.dumps(
        json,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    outbound_headers = dict(headers)
    outbound_headers["Host"] = target.host_header

    deadline = time.monotonic() + total_timeout
    raw_socket: socket.socket | None = None
    connection: http.client.HTTPConnection | None = None
    request_started = False
    try:
        family = socket.AF_INET6 if target.address.version == 6 else socket.AF_INET
        raw_socket = socket.socket(family, socket.SOCK_STREAM)
        raw_socket.settimeout(min(5.0, total_timeout))
        endpoint = (
            (str(target.address), target.port, 0, 0)
            if family == socket.AF_INET6
            else (str(target.address), target.port)
        )
        raw_socket.connect(endpoint)

        remaining = max(0.001, deadline - time.monotonic())
        raw_socket.settimeout(remaining)
        if target.scheme == "https":
            tls_context = ssl.create_default_context()
            raw_socket = tls_context.wrap_socket(
                raw_socket,
                server_hostname=target.hostname,
            )
            raw_socket.settimeout(max(0.001, deadline - time.monotonic()))

        connection = http.client.HTTPConnection(
            target.hostname,
            target.port,
            timeout=max(0.001, deadline - time.monotonic()),
        )
        connection.sock = raw_socket
        request_started = True
        connection.request(
            "POST",
            target.request_target,
            body=body,
            headers=outbound_headers,
        )
        raw_socket.settimeout(max(0.001, deadline - time.monotonic()))
        response = connection.getresponse()
        return _PinnedResponse(response, connection)
    except socket.timeout as exc:
        _close_transport(connection, raw_socket)
        if request_started:
            raise requests.exceptions.ReadTimeout() from exc
        raise requests.exceptions.ConnectTimeout() from exc
    except ssl.SSLCertVerificationError as exc:
        _close_transport(connection, raw_socket)
        raise _WebhookPolicyError("tls_certificate_invalid") from exc
    except ssl.SSLError as exc:
        _close_transport(connection, raw_socket)
        raise requests.exceptions.SSLError() from exc
    except http.client.InvalidURL as exc:
        _close_transport(connection, raw_socket)
        raise requests.exceptions.InvalidURL() from exc
    except (http.client.HTTPException, ConnectionError, OSError) as exc:
        _close_transport(connection, raw_socket)
        raise requests.exceptions.ConnectionError() from exc


def _close_transport(
    connection: http.client.HTTPConnection | None,
    raw_socket: socket.socket | None,
) -> None:
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass
        return
    if raw_socket is not None:
        try:
            raw_socket.close()
        except Exception:
            pass


def _resolve_addresses(
    hostname: str,
    port: int,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    direct_host = hostname.split("%", 1)[0]
    try:
        return (ipaddress.ip_address(direct_host),)
    except ValueError:
        pass

    resolved: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for family, _socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(
        hostname,
        port,
        type=socket.SOCK_STREAM,
    ):
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        raw_address = str(sockaddr[0]).split("%", 1)[0]
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            continue
        if address not in resolved:
            resolved.append(address)
    if not resolved:
        raise socket.gaierror(socket.EAI_NONAME, "hostname did not resolve")
    return tuple(resolved)


def _is_non_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        or not address.is_global
    )


def _validated_headers(configured: object) -> dict[str, str]:
    if not isinstance(configured, dict):
        raise _WebhookPolicyError("invalid_headers")
    headers: dict[str, str] = {}
    for raw_name, raw_value in configured.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or not raw_name.isascii()
            or raw_name != raw_name.strip()
            or not all(
                character.isalnum() or character in "!#$%&'*+-.^_`|~"
                for character in raw_name
            )
        ):
            raise _WebhookPolicyError("invalid_headers")
        if raw_name.lower() in _FORBIDDEN_REQUEST_HEADERS:
            raise _WebhookPolicyError("forbidden_header")
        if (
            not isinstance(raw_value, str)
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_value)
        ):
            raise _WebhookPolicyError("invalid_headers")
        try:
            raw_value.encode("latin-1")
        except UnicodeEncodeError:
            raise _WebhookPolicyError("invalid_headers") from None
        headers[raw_name] = raw_value
    return headers


def _close_response(response: object) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # A streamed response is already classified by this point. A
            # cleanup failure must not turn a provider acceptance into a retry.
            logger.warning("Webhook response cleanup failed after classification")
