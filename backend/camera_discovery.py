"""Network camera discovery, ONVIF enrichment, and RTSP validation helpers."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import secrets
import socket
import subprocess
import textwrap
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import cv2
import requests

from capability_registry import CAPABILITY_REGISTRY
from camera_connection import (
    DEFAULT_ONVIF_PORT,
    DEFAULT_RTSP_PORT,
    build_rtsp_url,
    infer_preferred_stream,
    normalize_stream_path,
    parse_rtsp_url,
)

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;3000000")

WS_DISCOVERY_ADDRESS = ("239.255.255.250", 3702)
WS_DISCOVERY_NS = {
    "e": "http://www.w3.org/2003/05/soap-envelope",
    "a": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
    "d": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
    "dn": "http://www.onvif.org/ver10/network/wsdl",
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "trt": "http://www.onvif.org/ver10/media/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
    "wsse": "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd",
    "wsu": "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd",
}
COMMON_RTSP_PORTS = [554, 8554, 10554]
CURATED_STREAM_PATHS = {
    "main": [
        "/Streaming/Channels/101",
        "/Streaming/Channels/1",
        "/cam/realmonitor?channel=1&subtype=0",
        "/h264Preview_01_main",
        "/unicaststream/1",
        "/stream1",
        "/live",
        "/live.sdp",
    ],
    "sub": [
        "/Streaming/Channels/102",
        "/Streaming/Channels/2",
        "/cam/realmonitor?channel=1&subtype=1",
        "/h264Preview_01_sub",
        "/unicaststream/2",
        "/stream2",
        "/sub",
        "/substream",
    ],
    "mjpeg": [
        "/unicaststream/3",
        "/mjpeg",
        "/video.mjpg",
    ],
}

DEFAULT_DISCOVERY_CAPABILITIES = ["person_presence"]
DEFAULT_DISCOVERY_ZONE = "Discovered Cameras"
DEFAULT_DISCOVERY_STREAM_PATH = "/Streaming/Channels/101"
DEFAULT_DISCOVERY_WINDOW = {
    "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    "from": "00:00",
    "to": "23:59",
}
DEFAULT_DISCOVERY_EVENT_OUTPUT_IDS = ["in_app"]
DEFAULT_DISCOVERY_EVENT_MESSAGE = "{severity} {violation_type} on {camera} in {zone}"


def _xml_text(parent: ET.Element | None, path: str, default: str = "") -> str:
    if parent is None:
        return default
    node = parent.find(path, WS_DISCOVERY_NS)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _extract_scope_value(scopes: str, key: str) -> str:
    marker = f"onvif://www.onvif.org/{key}/"
    for raw_scope in scopes.split():
        if raw_scope.startswith(marker):
            return raw_scope[len(marker):].replace("%20", " ")
    return ""


def _private_networks_only(networks: list[ipaddress._BaseNetwork]) -> list[ipaddress._BaseNetwork]:
    private_networks: list[ipaddress._BaseNetwork] = []
    seen: set[str] = set()
    for network in networks:
        if not network.is_private:
            continue
        serialized = str(network)
        if serialized in seen:
            continue
        private_networks.append(network)
        seen.add(serialized)
    return private_networks


def _discover_local_cidrs() -> list[ipaddress.IPv4Network]:
    networks: list[ipaddress.IPv4Network] = []
    try:
        result = subprocess.run(
            ["ip", "-o", "-f", "inet", "addr", "show", "scope", "global"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if "inet" not in parts:
                continue
            address = parts[parts.index("inet") + 1]
            try:
                networks.append(ipaddress.ip_network(address, strict=False))
            except ValueError:
                continue
    except Exception:
        pass

    if not networks:
        try:
            for _family, _socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = sockaddr[0]
                if ip.startswith("127."):
                    continue
                networks.append(ipaddress.ip_network(f"{ip}/24", strict=False))
        except Exception:
            pass

    return [network for network in _private_networks_only(networks) if isinstance(network, ipaddress.IPv4Network)]


def resolve_scan_networks(cidrs: list[str] | None = None) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if cidrs:
        networks: list[ipaddress.IPv4Network] = []
        for raw_cidr in cidrs:
            try:
                networks.append(ipaddress.ip_network(raw_cidr, strict=False))
            except ValueError:
                warnings.append(f"Ignored invalid CIDR: {raw_cidr}")
        return [str(network) for network in _private_networks_only(networks)], warnings

    networks = _discover_local_cidrs()
    if not networks:
        warnings.append("Could not infer a private subnet from the server interfaces.")
    return [str(network) for network in networks], warnings


def _probe_message() -> bytes:
    message_id = f"uuid:{uuid.uuid4()}"
    body = textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
                    xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
                    xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
                    xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
          <e:Header>
            <w:MessageID>{message_id}</w:MessageID>
            <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
            <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
          </e:Header>
          <e:Body>
            <d:Probe>
              <d:Types>dn:NetworkVideoTransmitter</d:Types>
            </d:Probe>
          </e:Body>
        </e:Envelope>
        """
    )
    return body.encode("utf-8")


def ws_discover(timeout_seconds: float = 4.0) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    seen: set[str] = set()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.settimeout(timeout_seconds)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    try:
        sock.sendto(_probe_message(), WS_DISCOVERY_ADDRESS)
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(64 * 1024)
            except socket.timeout:
                break
            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                continue
            probe_match = root.find(".//d:ProbeMatches/d:ProbeMatch", WS_DISCOVERY_NS)
            if probe_match is None:
                continue
            xaddrs = _xml_text(probe_match, "d:XAddrs")
            endpoint = _xml_text(probe_match, "a:EndpointReference/a:Address")
            scopes = _xml_text(probe_match, "d:Scopes")
            primary_xaddr = xaddrs.split()[0] if xaddrs else ""
            parsed_xaddr = urlparse(primary_xaddr) if primary_xaddr else None
            host = parsed_xaddr.hostname if parsed_xaddr else ""
            if not host:
                continue
            fingerprint = endpoint or f"onvif:{host}"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            devices.append({
                "fingerprint": endpoint or f"onvif:{host}",
                "source": "onvif",
                "ip": host,
                "host": host,
                "name": _extract_scope_value(scopes, "name") or host,
                "vendor": _extract_scope_value(scopes, "hardware"),
                "model": _extract_scope_value(scopes, "hardware"),
                "location": _extract_scope_value(scopes, "location"),
                "onvif_uuid": endpoint or None,
                "onvif_xaddr": primary_xaddr or None,
                "onvif_port": parsed_xaddr.port or DEFAULT_ONVIF_PORT if parsed_xaddr else None,
                "rtsp_port": None,
                "stream_candidates": [],
                "recommended_stream": "main",
                "auth_state": "unknown",
            })
    finally:
        sock.close()
    return devices


def _iter_candidate_hosts(network: str, *, max_hosts: int = 256) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    net = ipaddress.ip_network(network, strict=False)
    hosts = [str(host) for host in net.hosts()]
    if len(hosts) > max_hosts:
        warnings.append(f"Skipped hosts beyond the first {max_hosts} in {network}. Use a narrower CIDR for a full scan.")
        hosts = hosts[:max_hosts]
    return hosts, warnings


def _tcp_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _rtsp_options_auth_state(host: str, port: int, timeout: float = 0.5) -> str:
    request = (
        f"OPTIONS rtsp://{host}:{port}/ RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "User-Agent: Rakshak Lens/1.0\r\n\r\n"
    ).encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(request)
            response = sock.recv(1024).decode("utf-8", errors="ignore")
    except OSError:
        return "unknown"
    if "401" in response:
        return "needs_credentials"
    if "200" in response:
        return "valid"
    return "unknown"


def discover_cameras(cidrs: list[str] | None = None, timeout_seconds: float = 4.0) -> dict[str, Any]:
    networks, warnings = resolve_scan_networks(cidrs)
    try:
        onvif_devices = ws_discover(timeout_seconds=min(timeout_seconds, 4.0))
    except OSError as exc:
        onvif_devices = []
        warnings.append(f"ONVIF WS-Discovery failed: {exc}")
    devices_by_host: dict[str, dict[str, Any]] = {device["host"]: device for device in onvif_devices}

    rtsp_candidates: list[tuple[str, int]] = []
    for network in networks:
        hosts, host_warnings = _iter_candidate_hosts(network)
        warnings.extend(host_warnings)
        for host in hosts:
            for port in COMMON_RTSP_PORTS:
                rtsp_candidates.append((host, port))

    if rtsp_candidates:
        with ThreadPoolExecutor(max_workers=48) as pool:
            futures = {
                pool.submit(_tcp_port_open, host, port): (host, port)
                for host, port in rtsp_candidates
            }
            for future in as_completed(futures):
                host, port = futures[future]
                try:
                    is_open = future.result()
                except Exception:
                    is_open = False
                if not is_open:
                    continue
                device = devices_by_host.setdefault(host, {
                    "fingerprint": f"rtsp:{host}:{port}",
                    "source": "rtsp_probe",
                    "ip": host,
                    "host": host,
                    "name": host,
                    "vendor": "",
                    "model": "",
                    "location": "",
                    "onvif_uuid": None,
                    "onvif_xaddr": None,
                    "onvif_port": None,
                    "rtsp_port": port,
                    "stream_candidates": [],
                    "recommended_stream": "main",
                    "auth_state": "unknown",
                })
                if not device.get("rtsp_port"):
                    device["rtsp_port"] = port
                device["auth_state"] = _rtsp_options_auth_state(host, port)

    devices = sorted(devices_by_host.values(), key=lambda item: (item.get("name") or item["host"], item["host"]))
    return {
        "cidrs": networks,
        "warnings": warnings,
        "devices": devices,
    }


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "camera"


def _discovered_camera_id(device: dict[str, Any], index: int, used: set[str]) -> str:
    host = str(device.get("host") or device.get("ip") or "").strip()
    name = str(device.get("name") or "camera").strip()
    base = _slug(f"{name}_{host.replace('.', '_')}" if host else f"{name}_{index}")
    camera_id = f"discovered_{base}"
    suffix = 2
    while camera_id in used:
        camera_id = f"discovered_{base}_{suffix}"
        suffix += 1
    used.add(camera_id)
    return camera_id


def _stream_path_from_device(device: dict[str, Any], default_stream_path: str) -> tuple[str, bool]:
    explicit = normalize_stream_path(device.get("stream_path", ""))
    if explicit:
        return explicit, False
    for candidate in device.get("stream_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        path = normalize_stream_path(candidate.get("path", ""))
        if path:
            return path, False
    return normalize_stream_path(default_stream_path), True


def _safety_rule_ids_for(capabilities: list[str], explicit_rule_ids: list[str] | None = None) -> list[str]:
    if explicit_rule_ids is not None:
        return [str(rule_id) for rule_id in explicit_rule_ids if str(rule_id).strip()]
    rule_ids: list[str] = []
    for capability in capabilities:
        definition = CAPABILITY_REGISTRY.get(capability)
        if not definition:
            continue
        for rule_id in definition.get("safety_rule_ids", []):
            if rule_id not in rule_ids:
                rule_ids.append(rule_id)
    return rule_ids


def discovery_to_site_document(
    discovery: dict[str, Any],
    *,
    site_name: str = "Rakshak Lens Discovered Cameras",
    timezone: str = "Asia/Kolkata",
    merge_existing: bool = True,
    zone: str = DEFAULT_DISCOVERY_ZONE,
    profile: str = "general_safety",
    capabilities: list[str] | None = None,
    capability_model_overrides: dict[str, str] | None = None,
    safety_rule_ids: list[str] | None = None,
    username_env: str = "",
    password_env: str = "",
    default_stream_path: str = DEFAULT_DISCOVERY_STREAM_PATH,
    fps: int = 6,
    enabled: bool = False,
    include_active_windows: bool = True,
    include_event_policy: bool = True,
    event_output_ids: list[str] | None = None,
    event_severity: str = "inherit",
    event_priority: int = 5,
    event_cooldown_seconds: int = 60,
    event_min_confidence: float = 0.0,
    event_message_template: str = DEFAULT_DISCOVERY_EVENT_MESSAGE,
) -> dict[str, Any]:
    """Convert discovery output into a reviewable site YAML document.

    The generated cameras are disabled by default because RTSP discovery can
    find ports before it proves a readable stream path. Operators can pass
    enabled=True once credentials and stream paths have been verified.
    """
    selected_capabilities = [str(item) for item in (capabilities or DEFAULT_DISCOVERY_CAPABILITIES) if str(item).strip()]
    selected_overrides = {
        str(capability): str(model_key)
        for capability, model_key in (capability_model_overrides or {}).items()
        if str(capability).strip() and str(model_key).strip()
    }
    selected_rule_ids = _safety_rule_ids_for(selected_capabilities, safety_rule_ids)
    selected_output_ids = [
        str(item)
        for item in (event_output_ids if event_output_ids is not None else DEFAULT_DISCOVERY_EVENT_OUTPUT_IDS)
        if str(item).strip()
    ]
    cameras: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()

    for index, device in enumerate(discovery.get("devices") or [], start=1):
        if not isinstance(device, dict):
            continue
        host = str(device.get("host") or device.get("ip") or "").strip()
        if not host:
            continue
        stream_path, guessed_stream_path = _stream_path_from_device(device, default_stream_path)
        camera: dict[str, Any] = {
            "name": str(device.get("name") or host),
            "zone": zone,
            "profile": profile,
            "stream_type": "rtsp",
            "host": host,
            "rtsp_port": int(device.get("rtsp_port") or DEFAULT_RTSP_PORT),
            "stream_path": stream_path,
            "preferred_stream": str(device.get("recommended_stream") or device.get("preferred_stream") or infer_preferred_stream(stream_path)),
            "enabled": bool(enabled),
            "fps": int(fps),
            "capabilities": selected_capabilities,
            "capability_model_overrides": {
                capability: model_key
                for capability, model_key in selected_overrides.items()
                if capability in selected_capabilities
            },
            "safety_rule_ids": selected_rule_ids,
            "discovery_fingerprint": str(device.get("fingerprint") or ""),
            "onvif_uuid": device.get("onvif_uuid"),
            "onvif_xaddr": device.get("onvif_xaddr"),
            "onvif_port": device.get("onvif_port"),
            "vendor": device.get("vendor") or "",
            "model": device.get("model") or "",
            "discovery_source": device.get("source") or "discovery",
            "discovery_auth_state": device.get("auth_state") or "unknown",
            "needs_stream_path_review": guessed_stream_path,
        }
        if username_env:
            camera["username"] = f"${{{username_env}}}"
        if password_env:
            camera["password"] = f"${{{password_env}}}"
        if include_active_windows:
            camera["capability_windows"] = {
                capability: {
                    "mode": "detection",
                    "windows": [dict(DEFAULT_DISCOVERY_WINDOW)],
                }
                for capability in selected_capabilities
            }
        if include_event_policy and selected_output_ids:
            camera["event_policy"] = {
                "enabled": True,
                "output_ids": selected_output_ids,
                "severity": str(event_severity or "inherit"),
                "priority": max(1, int(event_priority)),
                "cooldown_seconds": max(0, int(event_cooldown_seconds)),
                "min_confidence": max(0.0, min(1.0, float(event_min_confidence))),
                "message_template": event_message_template or DEFAULT_DISCOVERY_EVENT_MESSAGE,
                "schedule": {"windows": [dict(DEFAULT_DISCOVERY_WINDOW)]},
            }
        camera_id = _discovered_camera_id(device, index, used_ids)
        cameras[camera_id] = {key: value for key, value in camera.items() if value not in (None, "")}

    return {
        "site": {
            "name": site_name,
            "timezone": timezone,
            "config_source": "yaml",
            "merge_existing": bool(merge_existing),
        },
        "cameras": cameras,
    }


def _wsse_header(username: str, password: str) -> str:
    nonce_bytes = secrets.token_bytes(12)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    digest = hashlib.sha1(nonce_bytes + created.encode("utf-8") + password.encode("utf-8")).digest()
    return (
        '<wsse:Security soap:mustUnderstand="1" '
        'xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" '
        'xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
        "<wsse:UsernameToken>"
        f"<wsse:Username>{username}</wsse:Username>"
        '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
        'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f"{base64.b64encode(digest).decode('ascii')}</wsse:Password>"
        '<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/'
        'oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
        f"{base64.b64encode(nonce_bytes).decode('ascii')}</wsse:Nonce>"
        f"<wsu:Created>{created}</wsu:Created>"
        "</wsse:UsernameToken></wsse:Security>"
    )


def _soap_envelope(body: str, username: str = "", password: str = "") -> str:
    security = _wsse_header(username, password) if username and password else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">'
        f"<soap:Header>{security}</soap:Header>"
        f"<soap:Body>{body}</soap:Body>"
        "</soap:Envelope>"
    )


def _soap_post(xaddr: str, body: str, username: str = "", password: str = "", timeout: float = 4.0) -> ET.Element:
    response = requests.post(
        xaddr,
        data=_soap_envelope(body, username, password),
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        timeout=timeout,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    fault = root.find(".//e:Fault", WS_DISCOVERY_NS)
    if fault is not None:
        reason = _xml_text(fault, "e:Reason/e:Text") or "ONVIF request failed"
        raise ValueError(reason)
    return root


def _resolve_onvif_streams(onvif_xaddr: str, username: str, password: str) -> dict[str, Any]:
    capabilities_root = _soap_post(
        onvif_xaddr,
        '<tds:GetCapabilities xmlns:tds="http://www.onvif.org/ver10/device/wsdl"><tds:Category>All</tds:Category></tds:GetCapabilities>',
        username=username,
        password=password,
    )
    media_xaddr = _xml_text(capabilities_root, ".//tt:Media/tt:XAddr")
    manufacturer = _xml_text(capabilities_root, ".//tds:GetDeviceInformationResponse/tds:Manufacturer")
    model = _xml_text(capabilities_root, ".//tds:GetDeviceInformationResponse/tds:Model")

    try:
        device_info_root = _soap_post(
            onvif_xaddr,
            '<tds:GetDeviceInformation xmlns:tds="http://www.onvif.org/ver10/device/wsdl" />',
            username=username,
            password=password,
        )
        manufacturer = _xml_text(device_info_root, ".//tds:Manufacturer") or manufacturer
        model = _xml_text(device_info_root, ".//tds:Model") or model
    except Exception:
        pass

    if not media_xaddr:
        raise ValueError("Media service not available")

    profiles_root = _soap_post(
        media_xaddr,
        '<trt:GetProfiles xmlns:trt="http://www.onvif.org/ver10/media/wsdl" />',
        username=username,
        password=password,
    )

    streams: list[dict[str, Any]] = []
    for profile in profiles_root.findall(".//trt:Profiles", WS_DISCOVERY_NS):
        token = profile.attrib.get("token", "")
        profile_name = profile.attrib.get("Name", "") or _xml_text(profile, "tt:Name")
        if not token:
            continue
        try:
            uri_root = _soap_post(
                media_xaddr,
                (
                    '<trt:GetStreamUri xmlns:trt="http://www.onvif.org/ver10/media/wsdl">'
                    '<trt:StreamSetup>'
                    '<tt:Stream xmlns:tt="http://www.onvif.org/ver10/schema">RTP-Unicast</tt:Stream>'
                    '<tt:Transport xmlns:tt="http://www.onvif.org/ver10/schema">'
                    "<tt:Protocol>RTSP</tt:Protocol>"
                    "</tt:Transport>"
                    "</trt:StreamSetup>"
                    f'<trt:ProfileToken>{token}</trt:ProfileToken>'
                    "</trt:GetStreamUri>"
                ),
                username=username,
                password=password,
            )
        except Exception:
            continue
        uri = _xml_text(uri_root, ".//tt:Uri")
        parsed_uri = parse_rtsp_url(uri)
        if not parsed_uri.get("host"):
            continue
        label = "main"
        lowered = profile_name.lower()
        if any(marker in lowered for marker in ("sub", "secondary", "low")):
            label = "sub"
        streams.append({
            "label": label,
            "name": profile_name or token,
            "path": normalize_stream_path(parsed_uri.get("stream_path", "")),
            "rtsp_port": parsed_uri.get("rtsp_port", DEFAULT_RTSP_PORT),
            "uri": uri,
        })

    return {
        "manufacturer": manufacturer,
        "model": model,
        "media_xaddr": media_xaddr,
        "streams": streams,
    }


def _ordered_curated_paths(preferred_stream: str) -> list[dict[str, str]]:
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    groups = [preferred_stream, "main", "sub", "mjpeg"]
    for group in groups:
        for path in CURATED_STREAM_PATHS.get(group, []):
            if path in seen:
                continue
            seen.add(path)
            ordered.append({"label": group, "name": group.title(), "path": normalize_stream_path(path)})
    return ordered


def _capture_preview(full_rtsp_url: str) -> dict[str, Any]:
    started = time.time()
    cap = cv2.VideoCapture(full_rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok = False
    frame = None
    for _ in range(15):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.1)
    cap.release()
    if not ok or frame is None:
        return {"ok": False, "latency_ms": int((time.time() - started) * 1000)}

    if frame.shape[1] > 320:
        scale = 320.0 / frame.shape[1]
        frame = cv2.resize(frame, (320, max(1, int(frame.shape[0] * scale))))
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if not success:
        return {"ok": False, "latency_ms": int((time.time() - started) * 1000)}
    return {
        "ok": True,
        "latency_ms": int((time.time() - started) * 1000),
        "preview_data_url": f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('ascii')}",
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }


def test_camera_connection(device: dict[str, Any]) -> dict[str, Any]:
    host = device.get("host") or device.get("ip") or ""
    rtsp_port = int(device.get("rtsp_port") or DEFAULT_RTSP_PORT)
    username = device.get("username", "") or ""
    password = device.get("password", "") or ""
    preferred_stream = device.get("preferred_stream") or "main"
    custom_stream_path = normalize_stream_path(device.get("stream_path", ""))
    onvif_xaddr = device.get("onvif_xaddr") or ""

    stream_candidates: list[dict[str, Any]] = []
    onvif_metadata: dict[str, Any] = {}
    auth_state = "unknown"

    if onvif_xaddr and username and password:
        try:
            onvif_metadata = _resolve_onvif_streams(onvif_xaddr, username, password)
            for stream in onvif_metadata.get("streams", []):
                stream_candidates.append(stream)
            auth_state = "valid"
        except ValueError as exc:
            if "author" in str(exc).lower():
                auth_state = "failed"
            else:
                auth_state = "unknown"
        except Exception:
            auth_state = "unknown"

    if custom_stream_path:
        stream_candidates.insert(0, {
            "label": preferred_stream if preferred_stream != "custom" else infer_preferred_stream(custom_stream_path),
            "name": "Custom",
            "path": custom_stream_path,
            "rtsp_port": rtsp_port,
        })

    if not stream_candidates:
        stream_candidates.extend(_ordered_curated_paths(preferred_stream))

    base_auth_state = _rtsp_options_auth_state(host, rtsp_port) if host else "unknown"
    if auth_state == "unknown":
        auth_state = base_auth_state

    errors: list[str] = []
    for candidate in stream_candidates:
        candidate_port = int(candidate.get("rtsp_port") or rtsp_port)
        full_rtsp_url = build_rtsp_url(
            {
                "host": host,
                "rtsp_port": candidate_port,
                "stream_path": candidate.get("path", ""),
                "username": username,
                "password": password,
            },
            include_credentials=True,
        )
        preview = _capture_preview(full_rtsp_url)
        if preview.get("ok"):
            return {
                "ok": True,
                "auth_state": "valid" if username or password else auth_state,
                "host": host,
                "rtsp_port": candidate_port,
                "onvif_port": device.get("onvif_port"),
                "stream_path": candidate.get("path", ""),
                "preferred_stream": candidate.get("label") or preferred_stream,
                "stream_candidates": [
                    {"label": stream.get("label"), "name": stream.get("name"), "path": stream.get("path")}
                    for stream in stream_candidates
                ],
                "recommended_stream": candidate.get("label") or preferred_stream,
                "preview_data_url": preview.get("preview_data_url"),
                "latency_ms": preview.get("latency_ms"),
                "width": preview.get("width"),
                "height": preview.get("height"),
                "onvif_uuid": device.get("onvif_uuid"),
                "onvif_xaddr": onvif_xaddr or None,
                "vendor": onvif_metadata.get("manufacturer") or device.get("vendor", ""),
                "model": onvif_metadata.get("model") or device.get("model", ""),
                "name": device.get("name") or host,
                "discovery_fingerprint": device.get("fingerprint") or f"rtsp:{host}:{candidate_port}:{candidate.get('path', '')}",
                "rtsp_url": build_rtsp_url(
                    {"host": host, "rtsp_port": candidate_port, "stream_path": candidate.get("path", "")},
                    include_credentials=False,
                ),
            }
        errors.append(candidate.get("path", ""))

    error_code = "stream_unreachable"
    error = "Could not read a frame from the camera stream."
    if auth_state in {"needs_credentials", "failed"}:
        error_code = "auth_failed"
        error = "Authentication failed or the camera requires different credentials."
    elif not host:
        error_code = "missing_host"
        error = "No camera host was available for testing."
    return {
        "ok": False,
        "auth_state": auth_state,
        "error_code": error_code,
        "error": error,
        "host": host,
        "rtsp_port": rtsp_port,
        "stream_candidates": [
            {"label": stream.get("label"), "name": stream.get("name"), "path": stream.get("path")}
            for stream in stream_candidates
        ],
        "attempted_paths": errors,
        "onvif_uuid": device.get("onvif_uuid"),
        "onvif_xaddr": onvif_xaddr or None,
        "discovery_fingerprint": device.get("fingerprint") or f"rtsp:{host}:{rtsp_port}",
        "vendor": onvif_metadata.get("manufacturer") or device.get("vendor", ""),
        "model": onvif_metadata.get("model") or device.get("model", ""),
        "name": device.get("name") or host,
    }
