#!/usr/bin/env python3
"""Sales-readiness video evaluation runner for Rakshak Lens.

The runner deliberately configures scenarios only through site YAML and
records runtime evidence from the public HTTP surfaces used by the UI.
"""

from __future__ import annotations

import argparse
import base64
import email.policy
import hashlib
import hmac
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_MANIFEST = ROOT / "qa" / "video_eval" / "manifest.yaml"
QUEUE_ANALYTIC_TYPES = {
    "queue_count",
    "queue_snapshot",
    "queue_duration",
    "queue_density",
    "queue_wait_time",
}
OBSTRUCTION_ANALYTIC_TYPES = {
    "obstruction_count",
    "obstruction_snapshot",
    "route_obstruction",
    "obstruction_duration",
    "obstruction_severity",
}
KNOWN_ANALYTICS_EXPECTATION_TYPES = {
    "active_window",
    "any_class_count",
    "capability_active_window",
    "capability_schedule_suppression",
    "class_absent",
    "class_count",
    "class_count_absent",
    "class_count_any",
    "classes_absent",
    "detector_suppression",
    "model_invocation",
    "model_invocations",
    "no_plate_read",
    "object_dwell",
    "object_lifecycle",
    "object_removal",
    "occupancy",
    "person_count",
    "plate_read",
    "plate_read_absent",
    "unattended_object_dwell",
    *QUEUE_ANALYTIC_TYPES,
    *OBSTRUCTION_ANALYTIC_TYPES,
}
SEVERITY_RANKS = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def model_server_url_from_site_config(config_path: Path) -> str | None:
    try:
        doc = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return None
    settings = doc.get("model_server") or {}
    if not isinstance(settings, dict) or not settings.get("enabled"):
        return None
    url = str(settings.get("url") or "").strip().rstrip("/")
    if not url:
        return None
    if "://" not in url:
        url = f"http://{url}"
    return url


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WebhookCaptureServer:
    """Small local receiver used to prove webhook delivery in eval runs."""

    def __init__(self, host: str, port: int, path: str) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8") or "{}")
                except Exception:
                    payload = None

                event = {
                    "timestamp": utc_now(),
                    "path": self.path,
                    "headers": {key: value for key, value in self.headers.items()},
                    "body": raw.decode("utf-8", errors="replace"),
                    "json": payload,
                }
                with capture._lock:
                    capture.requests.append(event)

                if self.path != capture.path:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            requests = list(self.requests)
        payloads = [item.get("json") for item in requests if isinstance(item.get("json"), dict)]
        alert_payloads = [
            normalize_captured_alert(payload.get("alert")) for payload in payloads
            if isinstance(payload.get("alert"), dict)
        ]
        return {
            "ok": bool(requests),
            "url": f"http://{self.host}:{self.port}{self.path}",
            "request_count": len(requests),
            "alert_count": len(alert_payloads),
            "paths": sorted({str(item.get("path")) for item in requests}),
            "alerts": alert_payloads[:10],
        }


def normalize_captured_alert(alert: dict[str, Any]) -> dict[str, Any]:
    """Normalize webhook/speaker payloads enough for delivery assertions."""
    normalized = dict(alert)
    normalized.setdefault("policyId", normalized.get("policy_id"))
    normalized.setdefault("rule", normalized.get("violation_type"))
    return normalized


class SMTPCaptureServer:
    """Minimal local SMTP receiver used to prove email delivery in eval runs."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        capture = self

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                mail_from = ""
                recipients: list[str] = []
                data_lines: list[bytes] = []
                in_data = False
                self.request.sendall(b"220 rakshak-eval-smtp ESMTP\r\n")
                stream = self.request.makefile("rb")

                while True:
                    line = stream.readline()
                    if not line:
                        break
                    stripped = line.rstrip(b"\r\n")
                    upper = stripped.upper()
                    if in_data:
                        if stripped == b".":
                            raw = b"".join(data_lines)
                            with capture._lock:
                                capture.messages.append(_summarize_smtp_message(mail_from, recipients, raw))
                            data_lines = []
                            in_data = False
                            self.request.sendall(b"250 queued\r\n")
                        else:
                            if stripped.startswith(b".."):
                                stripped = stripped[1:]
                            data_lines.append(stripped + b"\n")
                        continue
                    if upper.startswith(b"EHLO") or upper.startswith(b"HELO"):
                        self.request.sendall(b"250-rakshak-eval-smtp\r\n250 HELP\r\n")
                    elif upper.startswith(b"MAIL FROM:"):
                        mail_from = stripped.split(b":", 1)[1].strip().decode("utf-8", errors="replace").strip("<>")
                        self.request.sendall(b"250 sender ok\r\n")
                    elif upper.startswith(b"RCPT TO:"):
                        recipient = stripped.split(b":", 1)[1].strip().decode("utf-8", errors="replace").strip("<>")
                        recipients.append(recipient)
                        self.request.sendall(b"250 recipient ok\r\n")
                    elif upper == b"DATA":
                        in_data = True
                        self.request.sendall(b"354 end with <CRLF>.<CRLF>\r\n")
                    elif upper == b"RSET":
                        mail_from = ""
                        recipients = []
                        data_lines = []
                        in_data = False
                        self.request.sendall(b"250 reset\r\n")
                    elif upper == b"NOOP":
                        self.request.sendall(b"250 ok\r\n")
                    elif upper == b"QUIT":
                        self.request.sendall(b"221 bye\r\n")
                        break
                    else:
                        self.request.sendall(b"250 ok\r\n")

        self._server = Server((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            messages = list(self.messages)
        return {
            "ok": bool(messages),
            "address": f"{self.host}:{self.port}",
            "message_count": len(messages),
            "messages": messages[:10],
        }


def _summarize_smtp_message(mail_from: str, recipients: list[str], raw: bytes) -> dict[str, Any]:
    parsed = BytesParser(policy=email.policy.default).parsebytes(raw)
    body = ""
    try:
        if parsed.is_multipart():
            for part in parsed.walk():
                if part.get_content_type() in {"text/plain", "text/html"}:
                    body += str(part.get_content() or "")
        else:
            body = str(parsed.get_content() or "")
    except Exception:
        body = raw.decode("utf-8", errors="replace")
    return {
        "timestamp": utc_now(),
        "mail_from": mail_from,
        "recipients": recipients,
        "subject": parsed.get("subject", ""),
        "from": parsed.get("from", ""),
        "to": parsed.get("to", ""),
        "body_preview": body[:1000],
        "bytes": len(raw),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a YAML object")
    return data


def find_scenario(manifest: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    for scenario in manifest.get("scenarios", []):
        if scenario.get("id") == scenario_id:
            return scenario
    raise KeyError(f"Scenario not found: {scenario_id}")


def run_command(args: list[str], *, timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def admin_token() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        cfg_path = BACKEND_DIR / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        secret = cfg.get("auth", {}).get("jwt_secret")
    if not secret:
        raise RuntimeError("JWT secret not found; cannot create eval admin token")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": "video-eval",
        "username": "video-eval",
        "role": "admin",
        "iat": now,
        "exp": now + 8 * 60 * 60,
    }
    signing_input = ".".join([_b64url_json(header), _b64url_json(payload)]).encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return signing_input.decode("ascii") + "." + _b64url(signature)


def _b64url_json(value: dict[str, Any]) -> str:
    return _b64url(json.dumps(value, separators=(",", ":")).encode("utf-8"))


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def request_json(url: str, token: str | None = None, *, timeout: float = 5.0) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.status, json.loads(raw.decode("utf-8") or "{}")
            return response.status, raw.decode("utf-8", errors="replace")[:500]
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = raw.decode("utf-8", errors="replace")[:500]
        return exc.code, body


def post_json(
    url: str,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.status, json.loads(raw.decode("utf-8") or "{}")
            return response.status, raw.decode("utf-8", errors="replace")[:500]
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = raw.decode("utf-8", errors="replace")[:500]
        return exc.code, body
    except Exception as exc:
        return 0, {"error": str(exc)}


def stream_probe(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            chunk = response.read(2048)
            return {
                "ok": response.status == 200 and bool(chunk),
                "status": response.status,
                "content_type": response.headers.get("content-type", ""),
                "bytes_read": len(chunk),
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def capture_stream_frame(url: str, path: Path, *, timeout: float = 6.0) -> dict[str, Any]:
    """Capture one JPEG frame from the MJPEG stream as visual evidence."""
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    buffer = b""
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            while time.time() - started < timeout:
                chunk = response.read(4096)
                if not chunk:
                    break
                buffer += chunk
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    buffer = buffer[-4096:]
                    continue
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    continue
                frame = buffer[start:end + 2]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(frame)
                return {
                    "ok": True,
                    "path": str(path.relative_to(ROOT)),
                    "bytes_written": len(frame),
                    "content_type": response.headers.get("content-type", ""),
                    "captured_at": utc_now(),
                }
        return {"ok": False, "error": "No complete JPEG frame found in stream"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_camera(cameras_payload: Any, camera_id: str) -> dict[str, Any] | None:
    if isinstance(cameras_payload, list):
        cameras = cameras_payload
    elif isinstance(cameras_payload, dict):
        cameras = cameras_payload.get("cameras", [])
    else:
        cameras = []
    for camera in cameras:
        if isinstance(camera, dict) and camera.get("id") == camera_id:
            return camera
    return None


def required_model_keys_from_camera(camera: dict[str, Any] | None) -> list[str]:
    if not isinstance(camera, dict):
        return []
    plan = camera.get("execution_plan") or camera.get("executionPlan") or {}
    if not isinstance(plan, dict):
        return []
    raw = plan.get("required_model_keys") or plan.get("requiredModelKeys") or []
    if not isinstance(raw, list):
        return []
    return [str(model_key) for model_key in raw if isinstance(model_key, str) and model_key]


def model_readiness_from_health(health: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(health, dict):
        return {}
    models = health.get("models") or []
    if not isinstance(models, list):
        return {}
    readiness: dict[str, dict[str, Any]] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        model_key = model.get("model_key") or model.get("modelKey")
        if not model_key:
            continue
        status = str(model.get("status") or "")
        readiness[str(model_key)] = {
            "status": status,
            "is_ready": bool(model.get("is_ready") or model.get("isReady") or status == "ready"),
            "active_path": model.get("active_path") or model.get("activePath"),
            "error": model.get("error"),
        }
    return readiness


def model_preflight_status(camera: dict[str, Any] | None, health: dict[str, Any] | None) -> dict[str, Any]:
    required_model_keys = required_model_keys_from_camera(camera)
    readiness = model_readiness_from_health(health)
    if not isinstance(camera, dict):
        return {
            "checked": False,
            "ok": None,
            "reason": "camera_not_available",
            "required_model_keys": required_model_keys,
            "missing_model_keys": [],
            "models": readiness,
        }
    if not isinstance(health, dict) or not readiness:
        return {
            "checked": False,
            "ok": None,
            "reason": "health_model_snapshot_not_available",
            "required_model_keys": required_model_keys,
            "missing_model_keys": [],
            "models": readiness,
        }
    missing_model_keys = [
        model_key
        for model_key in required_model_keys
        if readiness.get(model_key, {}).get("is_ready") is not True
    ]
    return {
        "checked": True,
        "ok": not missing_model_keys,
        "reason": "ok" if not missing_model_keys else "required_models_not_ready",
        "required_model_keys": required_model_keys,
        "missing_model_keys": missing_model_keys,
        "models": {model_key: readiness.get(model_key, {"status": "missing", "is_ready": False}) for model_key in required_model_keys},
    }


def classify_result(result: dict[str, Any]) -> str:
    if result.get("blocking_errors"):
        return "blocked"
    evidence = result.get("evidence", {})
    camera = evidence.get("final_camera") or {}
    alerts = evidence.get("matching_alerts") or []
    stream = evidence.get("stream_probe") or {}
    expected_output_ids = set(evidence.get("expected_output_ids") or [])
    ui = evidence.get("ui_evidence") or {}
    ui_screenshot_required = bool(ui.get("screenshot_required"))
    alerts_required = bool(evidence.get("expected_alerts_required", True))
    detections_observed = evidence.get("max_detections_count", 0) > 0
    detections_verified = detections_observed or ui.get("detections_count_should_increase") is False
    alerts_verified = bool(alerts) if alerts_required else True
    outputs_verified = not expected_output_ids or alerts_cover_outputs(alerts, expected_output_ids)
    delivery_verified = delivery_expectations_pass(evidence)
    ui_verified = not ui_screenshot_required or bool(ui.get("screenshot_fresh"))
    analytics_verified = analytics_expectations_pass(evidence)
    unexpected_alerts_clear = not evidence.get("unexpected_alerts")
    if (
        camera.get("runtime_status") == "running"
        and camera.get("status") == "online"
        and stream.get("ok")
        and detections_verified
        and alerts_verified
        and outputs_verified
        and delivery_verified
        and ui_verified
        and analytics_verified
        and unexpected_alerts_clear
    ):
        return "ready_to_sell"
    if camera.get("runtime_status") in {"running", "starting"} or stream.get("ok") or detections_observed:
        return "needs_work"
    return "blocked"


def analytics_expectations_pass(evidence: dict[str, Any]) -> bool:
    expectations = evidence.get("expected_analytics") or []
    if not expectations:
        return True
    if unsupported_analytics_expectation_types(expectations):
        return False

    summary = evidence.get("analytics_summary") or {}
    for expectation in expectations:
        expectation_type = expectation.get("type")
        if expectation_type == "occupancy":
            occupancy = summary.get("occupancy") or {}
            if not occupancy.get("ok"):
                return False
            if occupancy.get("zone_count", 0) < int(expectation.get("min_zone_count", 0)):
                return False
            if occupancy.get("occupied_count", 0) < int(expectation.get("min_occupied_zones", 0)):
                return False
            if occupancy.get("sample_seconds", 0) < int(expectation.get("min_sample_seconds", 0)):
                return False
            if occupancy.get("max_occupied_seconds", 0) < int(expectation.get("min_max_occupied_seconds", 0)):
                return False
            if occupancy.get("max_longest_occupied_seconds", 0) < int(expectation.get("min_longest_occupied_seconds", 0)):
                return False
            if occupancy.get("duration_ready_zone_count", 0) < int(expectation.get("min_duration_ready_zones", 0)):
                return False
            if expectation.get("duration_ready") is True and not occupancy.get("duration_ready"):
                return False
            continue
        if expectation_type == "plate_read":
            plate_reads = summary.get("plate_reads") or {}
            if not plate_reads.get("ok"):
                return False
            if plate_reads.get("read_count", 0) < int(expectation.get("min_count", 0)):
                return False
            expected_plate = expectation.get("plate") or expectation.get("normalized_plate")
            if expected_plate and normalize_plate_text(expected_plate) not in plate_reads.get("normalized_plates", []):
                return False
            expected_event_types = set(expectation.get("event_types") or [])
            if expected_event_types and not expected_event_types.intersection(plate_reads.get("event_types", [])):
                return False
            min_confidence = expectation.get("min_confidence")
            if min_confidence is not None and float(plate_reads.get("max_confidence") or 0) < float(min_confidence):
                return False
            continue
        if expectation_type in {"plate_read_absent", "no_plate_read"}:
            plate_reads = summary.get("plate_reads") or {}
            max_count = int(expectation.get("max_count", 0))
            if int(plate_reads.get("read_count") or 0) > max_count:
                return False
            expected_plate = expectation.get("plate") or expectation.get("normalized_plate")
            if expected_plate and normalize_plate_text(expected_plate) in plate_reads.get("normalized_plates", []):
                return False
            forbidden_event_types = set(expectation.get("event_types") or [])
            if forbidden_event_types and forbidden_event_types.intersection(plate_reads.get("event_types", [])):
                return False
            continue
        if expectation_type in {"person_count", "class_count"}:
            class_name = expectation.get("class", "person")
            class_counts = summary.get("class_counts") or {}
            observed = int(class_counts.get(class_name, 0))
            if observed < int(expectation.get("min_count", 0)):
                return False
            continue
        if expectation_type in {"class_absent", "classes_absent", "class_count_absent"}:
            class_counts = summary.get("class_counts") or {}
            class_names = [str(item) for item in expectation.get("classes", [])]
            if not class_names and expectation.get("class"):
                class_names = [str(expectation["class"])]
            max_count = int(expectation.get("max_count", 0))
            if any(int(class_counts.get(class_name, 0)) > max_count for class_name in class_names):
                return False
            continue
        if expectation_type in {"detector_suppression", "capability_schedule_suppression"}:
            schedule = summary.get("schedule") or {}
            if not schedule.get("ok"):
                return False
            capability = expectation.get("capability")
            if capability and capability not in schedule.get("suppressed_capabilities", []):
                return False
            max_detections = expectation.get("max_detections")
            if max_detections is not None and int(evidence.get("max_detections_count") or 0) > int(max_detections):
                return False
            class_counts = summary.get("class_counts") or {}
            for class_name in expectation.get("classes_absent") or []:
                if int(class_counts.get(str(class_name), 0)) > 0:
                    return False
            model_invocations = schedule.get("model_invocations") or {}
            max_model_invocations = expectation.get("max_model_invocations")
            if max_model_invocations is not None:
                for model_key in expectation.get("model_keys") or []:
                    if int(model_invocations.get(str(model_key), 0)) > int(max_model_invocations):
                        return False
            continue
        if expectation_type in {"active_window", "capability_active_window"}:
            schedule = summary.get("schedule") or {}
            if not schedule.get("ok"):
                return False
            capability = expectation.get("capability")
            if capability and capability in schedule.get("suppressed_capabilities", []):
                return False
            model_invocations = schedule.get("model_invocations") or {}
            model_keys = [str(model_key) for model_key in expectation.get("model_keys") or []]
            min_model_invocations = int(expectation.get("min_model_invocations", 1))
            if model_keys:
                for model_key in model_keys:
                    if int(model_invocations.get(model_key, 0)) < min_model_invocations:
                        return False
            elif not any(int(count or 0) >= min_model_invocations for count in model_invocations.values()):
                return False
            continue
        if expectation_type in {"model_invocation", "model_invocations"}:
            schedule = summary.get("schedule") or {}
            if not schedule.get("ok"):
                return False
            model_invocations = schedule.get("model_invocations") or {}
            model_keys = [str(model_key) for model_key in expectation.get("model_keys") or []]
            min_model_invocations = int(expectation.get("min_model_invocations", 1))
            max_model_invocations = expectation.get("max_model_invocations")
            if model_keys:
                for model_key in model_keys:
                    observed = int(model_invocations.get(model_key, 0))
                    if observed < min_model_invocations:
                        return False
                    if max_model_invocations is not None and observed > int(max_model_invocations):
                        return False
            elif not any(int(count or 0) >= min_model_invocations for count in model_invocations.values()):
                return False
            continue
        if expectation_type in {"class_count_any", "any_class_count"}:
            class_names = [str(item) for item in expectation.get("classes", [])]
            if not class_names and expectation.get("class"):
                class_names = [str(expectation["class"])]
            class_counts = summary.get("class_counts") or {}
            observed = sum(int(class_counts.get(name, 0)) for name in class_names)
            if observed < int(expectation.get("min_count", 0)):
                return False
            continue
        if expectation_type in QUEUE_ANALYTIC_TYPES:
            queue = summary.get("queue") or {}
            if not queue.get("ok"):
                return False
            if int(queue.get("person_count", 0)) < int(expectation.get("min_count", 0)):
                return False
            if int(queue.get("max_zone_count", 0)) < int(expectation.get("min_zone_count", 0)):
                return False
            if expectation.get("queue_active") is True and not queue.get("queue_active"):
                return False
            if int(queue.get("active_seconds", 0)) < int(expectation.get("min_active_seconds", 0)):
                return False
            if int(queue.get("longest_active_seconds", 0)) < int(expectation.get("min_longest_active_seconds", 0)):
                return False
            if expectation.get("duration_ready") is True and not queue.get("duration_ready"):
                return False
            if expectation_type == "queue_density":
                if expectation.get("calibrated", True) and not queue.get("calibrated"):
                    return False
                if int(queue.get("calibrated_zone_count", 0)) < int(expectation.get("min_calibrated_zone_count", 1)):
                    return False
                min_density = expectation.get("min_density_people_per_square_meter")
                if min_density is not None and float(queue.get("max_density_people_per_square_meter") or 0) < float(min_density):
                    return False
                max_density = expectation.get("max_density_people_per_square_meter")
                if max_density is not None and float(queue.get("max_density_people_per_square_meter") or 0) > float(max_density):
                    return False
            if expectation_type == "queue_wait_time":
                if expectation.get("wait_tracking_enabled", True) and not queue.get("wait_tracking_enabled"):
                    return False
                if int(queue.get("tracked_person_count", 0)) < int(expectation.get("min_tracked_person_count", 1)):
                    return False
                if int(queue.get("max_wait_seconds", 0)) < int(expectation.get("min_max_wait_seconds", 0)):
                    return False
                if expectation.get("wait_time_ready") is True and not queue.get("wait_time_ready"):
                    return False
            continue
        if expectation_type in OBSTRUCTION_ANALYTIC_TYPES:
            obstruction = summary.get("obstruction") or {}
            if not obstruction.get("ok"):
                return False
            if int(obstruction.get("object_count", 0)) < int(expectation.get("min_count", 0)):
                return False
            max_count = expectation.get("max_count")
            if max_count is not None and int(obstruction.get("object_count", 0)) > int(max_count):
                return False
            if int(obstruction.get("max_zone_count", 0)) < int(expectation.get("min_zone_count", 0)):
                return False
            max_zone_count = expectation.get("max_zone_count")
            if max_zone_count is not None and int(obstruction.get("max_zone_count", 0)) > int(max_zone_count):
                return False
            if expectation.get("obstruction_active") is True and not obstruction.get("obstruction_active"):
                return False
            if expectation.get("obstruction_active") is False and obstruction.get("obstruction_active"):
                return False
            if int(obstruction.get("active_seconds", 0)) < int(expectation.get("min_active_seconds", 0)):
                return False
            if int(obstruction.get("longest_active_seconds", 0)) < int(expectation.get("min_longest_active_seconds", 0)):
                return False
            if expectation.get("duration_ready") is True and not obstruction.get("duration_ready"):
                return False
            expected_classes = {str(item) for item in expectation.get("classes", [])}
            if expected_classes and not expected_classes.intersection(set(obstruction.get("observed_classes", []))):
                return False
            if expectation_type == "obstruction_severity":
                if expectation.get("calibrated", True) and not obstruction.get("calibrated"):
                    return False
                if int(obstruction.get("calibrated_zone_count", 0)) < int(expectation.get("min_calibrated_zone_count", 1)):
                    return False
                min_density = expectation.get("min_density_objects_per_square_meter")
                if min_density is not None and float(obstruction.get("max_density_objects_per_square_meter") or 0) < float(min_density):
                    return False
                min_severity = str(expectation.get("min_severity") or "none").lower()
                if SEVERITY_RANKS.get(str(obstruction.get("max_severity") or "none").lower(), 0) < SEVERITY_RANKS.get(min_severity, 0):
                    return False
            continue
        if expectation_type in {"object_lifecycle", "object_removal", "object_dwell", "unattended_object_dwell"}:
            lifecycle = summary.get("object_lifecycle") or {}
            if not lifecycle.get("ok"):
                return False
            if int(lifecycle.get("removal_count", 0)) < int(expectation.get("min_removal_count", 0)):
                return False
            if int(lifecycle.get("dwell_count", 0)) < int(expectation.get("min_dwell_count", 0)):
                return False
            if float(lifecycle.get("max_present_seconds", 0)) < float(expectation.get("min_present_seconds", 0)):
                return False
            if int(lifecycle.get("seen_zone_count", 0)) < int(expectation.get("min_seen_zones", 0)):
                return False
            if expectation.get("removal_detected") is True and not lifecycle.get("removal_detected"):
                return False
            if expectation.get("dwell_detected") is True and not lifecycle.get("dwell_detected"):
                return False
            if expectation.get("dwell_ready") is True and not lifecycle.get("dwell_ready"):
                return False
            expected_classes = {str(item) for item in expectation.get("classes", [])}
            if expected_classes and not expected_classes.intersection(set(lifecycle.get("observed_classes", []))):
                return False
            continue
        return False
    return True


def unsupported_analytics_expectation_types(expectations: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(expectation.get("type") or "")
            for expectation in expectations
            if isinstance(expectation, dict)
            and str(expectation.get("type") or "") not in KNOWN_ANALYTICS_EXPECTATION_TYPES
        }
    )


def delivery_expectations_pass(evidence: dict[str, Any]) -> bool:
    expected = evidence.get("expected_delivery") or {}
    webhook_expected = expected.get("webhook_capture") or {}
    if webhook_expected:
        capture = evidence.get("webhook_capture") or {}
        if not capture.get("ok"):
            return False
        if int(capture.get("request_count") or 0) < int(webhook_expected.get("min_requests", 1)):
            return False
        expected_policy = webhook_expected.get("policy_id")
        if expected_policy:
            alerts = capture.get("alerts") or []
            if not any(isinstance(alert, dict) and alert.get("policyId") == expected_policy for alert in alerts):
                return False
        expected_rule = webhook_expected.get("rule")
        if expected_rule:
            alerts = capture.get("alerts") or []
            if not any(isinstance(alert, dict) and alert.get("rule") == expected_rule for alert in alerts):
                return False
    smtp_expected = expected.get("smtp_capture") or {}
    if smtp_expected:
        capture = evidence.get("smtp_capture") or {}
        if not capture.get("ok"):
            return False
        if int(capture.get("message_count") or 0) < int(smtp_expected.get("min_messages", 1)):
            return False
        messages = [message for message in capture.get("messages") or [] if isinstance(message, dict)]
        subject_contains = smtp_expected.get("subject_contains")
        if subject_contains and not any(subject_contains in str(message.get("subject", "")) for message in messages):
            return False
        body_contains = smtp_expected.get("body_contains")
        if body_contains and not any(body_contains in str(message.get("body_preview", "")) for message in messages):
            return False
        recipient = smtp_expected.get("recipient")
        if recipient and not any(recipient in (message.get("recipients") or []) for message in messages):
            return False
    return True


def scenario_result_dir(artifact_root: Path, scenario: dict[str, Any]) -> Path:
    """Return the result directory for a scenario, optionally under results/<subdir>."""
    base_result_dir = artifact_root / "results"
    runtime = scenario.get("runtime") if isinstance(scenario.get("runtime"), dict) else {}
    raw_subdir = runtime.get("result_subdir") or scenario.get("result_subdir")
    if not raw_subdir:
        return base_result_dir
    subdir = Path(str(raw_subdir))
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("scenario result_subdir must be relative and stay inside results")
    return base_result_dir / subdir


def run_scenario(manifest_path: Path, scenario_id: str, *, skip_apply: bool = False) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    scenario = find_scenario(manifest, scenario_id)
    artifact_root = ROOT / manifest.get("artifact_root", "qa/video_eval")
    result_dir = scenario_result_dir(artifact_root, scenario)
    log_dir = artifact_root / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_config_path = (
        scenario.get("config_path")
        or (scenario.get("runtime", {}) or {}).get("config_path")
        or manifest.get("config_path", "qa/video_eval/site.yaml")
    )
    config_path = Path(str(raw_config_path))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    backend_url = scenario.get("runtime", {}).get("backend_url", "http://127.0.0.1:8000").rstrip("/")
    model_server_url = model_server_url_from_site_config(config_path)
    wait_seconds = int(scenario.get("runtime", {}).get("wait_seconds", 60))
    poll_interval = float(scenario.get("runtime", {}).get("poll_interval_seconds", 3))
    camera_id = scenario["camera_id"]
    expected_alerts = scenario.get("expected", {}).get("alerts", [])
    expected_alerts_absent = scenario.get("expected", {}).get("alerts_absent", [])
    expected_analytics = scenario.get("expected", {}).get("analytics", [])
    expected_delivery = scenario.get("expected", {}).get("delivery", {})
    expected_ui = scenario.get("expected", {}).get("ui_evidence", {})
    expected_rule_names = {item.get("rule") for item in expected_alerts if item.get("rule")}
    expected_policy_ids = {item.get("policy_id") for item in expected_alerts if item.get("policy_id")}
    expected_output_ids = {
        output_id
        for item in expected_alerts
        for output_id in (item.get("output_ids") or item.get("outputIds") or [])
    }
    forbidden_rule_names = {item.get("rule") for item in expected_alerts_absent if item.get("rule")}
    forbidden_policy_ids = {item.get("policy_id") for item in expected_alerts_absent if item.get("policy_id")}
    video_path = ROOT / str(scenario.get("local_video", ""))

    result: dict[str, Any] = {
        "scenario_id": scenario_id,
        "started_at": utc_now(),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "camera_id": camera_id,
        "video": scenario.get("local_video"),
        "video_sha256": sha256_file(video_path) if video_path.exists() else None,
        "source": scenario.get("source", {}),
        "yaml_commands": [],
        "runtime_reload": None,
        "polls": [],
        "evidence": {},
        "blocking_errors": [],
    }
    unsupported_analytics_types = unsupported_analytics_expectation_types(expected_analytics)
    if unsupported_analytics_types:
        result["blocking_errors"].append(
            "Unsupported analytics expectation type(s): "
            + ", ".join(unsupported_analytics_types)
        )
        result["unsupported_analytics_expectation_types"] = unsupported_analytics_types

    if not video_path.exists():
        result["blocking_errors"].append(f"Video file not found: {video_path}")

    if not config_path.exists():
        result["blocking_errors"].append(f"Site YAML not found: {config_path}")

    if result["blocking_errors"]:
        result["status"] = "blocked"
        result["finished_at"] = utc_now()
        write_result(result_dir, result)
        return result

    webhook_capture = None
    capture_cfg = scenario.get("runtime", {}).get("webhook_capture") or {}
    if capture_cfg.get("enabled"):
        try:
            webhook_capture = WebhookCaptureServer(
                str(capture_cfg.get("host", "127.0.0.1")),
                int(capture_cfg.get("port", 18080)),
                str(capture_cfg.get("path", "/rakshak-webhook")),
            )
            webhook_capture.start()
        except Exception as exc:
            result["blocking_errors"].append(f"Webhook capture server failed to start: {exc}")
            result["status"] = "blocked"
            result["finished_at"] = utc_now()
            write_result(result_dir, result)
            return result
    smtp_capture = None
    smtp_capture_cfg = scenario.get("runtime", {}).get("smtp_capture") or {}
    if smtp_capture_cfg.get("enabled"):
        try:
            smtp_capture = SMTPCaptureServer(
                str(smtp_capture_cfg.get("host", "127.0.0.1")),
                int(smtp_capture_cfg.get("port", 18081)),
            )
            smtp_capture.start()
        except Exception as exc:
            if webhook_capture:
                webhook_capture.stop()
            result["blocking_errors"].append(f"SMTP capture server failed to start: {exc}")
            result["status"] = "blocked"
            result["finished_at"] = utc_now()
            write_result(result_dir, result)
            return result

    yaml_commands = [
        [sys.executable, "scripts/safetylens_site.py", "--config", str(config_path), "validate"],
        [sys.executable, "scripts/safetylens_site.py", "--config", str(config_path), "plan"],
    ]
    if skip_apply:
        result["yaml_commands"].append(
            {
                "args": ["scripts/safetylens_site.py", "--config", str(config_path), "apply", "--yes"],
                "returncode": 0,
                "elapsed_seconds": 0,
                "stdout": "Skipped by --skip-apply; config must be pre-applied before backend startup.\n",
                "stderr": "",
                "skipped": True,
            }
        )
    else:
        yaml_commands.append([sys.executable, "scripts/safetylens_site.py", "--config", str(config_path), "apply", "--yes"])

    for command in yaml_commands:
        command_result = run_command(command)
        result["yaml_commands"].append(command_result)
        if command_result["returncode"] != 0:
            result["blocking_errors"].append(
                f"YAML command failed ({command_result['returncode']}): {' '.join(command)}"
            )
            result["status"] = "blocked"
            result["finished_at"] = utc_now()
            write_result(result_dir, result)
            if webhook_capture:
                webhook_capture.stop()
            if smtp_capture:
                smtp_capture.stop()
            return result

    token = admin_token()
    if skip_apply:
        result["runtime_reload"] = {
            "skipped": True,
            "reason": "--skip-apply assumes config was applied before backend startup",
        }
    else:
        reload_started = time.perf_counter()
        reload_status, reload_payload = post_json(f"{backend_url}/api/config/reload", token)
        result["runtime_reload"] = {
            "url": f"{backend_url}/api/config/reload",
            "status": reload_status,
            "elapsed_seconds": round(time.perf_counter() - reload_started, 3),
            "response": reload_payload,
            "ok": 200 <= reload_status < 300 and bool(isinstance(reload_payload, dict) and reload_payload.get("ok")),
        }
        if not result["runtime_reload"]["ok"]:
            result["blocking_errors"].append(f"Runtime config reload failed: {reload_payload}")
            result["status"] = "blocked"
            result["finished_at"] = utc_now()
            write_result(result_dir, result)
            if webhook_capture:
                webhook_capture.stop()
            if smtp_capture:
                smtp_capture.stop()
            return result
        time.sleep(float(scenario.get("runtime", {}).get("reload_wait_seconds", 1.0)))

    model_preflight = {
        "checked": False,
        "ok": None,
        "reason": "not_run",
        "required_model_keys": [],
        "missing_model_keys": [],
        "models": {},
    }
    try:
        preflight_health_status, preflight_health = request_json(f"{backend_url}/api/health")
        preflight_cameras_status, preflight_cameras = request_json(f"{backend_url}/api/cameras", token)
        preflight_camera = get_camera(preflight_cameras, camera_id)
        model_preflight = model_preflight_status(preflight_camera, preflight_health)
        model_preflight["health_status"] = preflight_health_status
        model_preflight["cameras_status"] = preflight_cameras_status
        if model_preflight.get("checked") and model_preflight.get("ok") is not True:
            missing = ", ".join(model_preflight.get("missing_model_keys") or [])
            result["blocking_errors"].append(f"Required models not ready: {missing}")
            result["evidence"] = {
                "health": preflight_health if isinstance(preflight_health, dict) else None,
                "final_camera": preflight_camera,
                "model_preflight": model_preflight,
            }
            result["status"] = "blocked"
            result["finished_at"] = utc_now()
            write_result(result_dir, result)
            if webhook_capture:
                webhook_capture.stop()
            if smtp_capture:
                smtp_capture.stop()
            return result
    except Exception as exc:
        model_preflight = {
            **model_preflight,
            "reason": f"preflight_request_failed:{exc}",
        }

    started = time.time()
    scenario_started_at = result["started_at"]
    max_detections = 0
    final_camera = None
    final_health = None
    final_model_server_health = None
    final_model_server_status = None
    final_alerts: list[dict[str, Any]] = []
    final_analytics: dict[str, Any] = {}
    max_detection_class_counts: dict[str, int] = {}

    while time.time() - started < wait_seconds:
        poll: dict[str, Any] = {"timestamp": utc_now()}
        try:
            health_status, health = request_json(f"{backend_url}/api/health")
            model_server_health_status = None
            model_server_health = None
            if model_server_url:
                model_server_health_status, model_server_health = request_json(f"{model_server_url}/api/health")
            cameras_status, cameras = request_json(f"{backend_url}/api/cameras", token)
            alerts_status, alerts = request_json(
                f"{backend_url}/api/alerts?cameraId={camera_id}&limit=50",
                token,
            )
            occupancy_status = None
            occupancy = None
            if any(item.get("type") == "occupancy" for item in expected_analytics):
                occupancy_status, occupancy = request_json(
                    f"{backend_url}/api/cameras/{camera_id}/occupancy",
                    token,
                )
            queue_status = None
            queue = None
            if any(item.get("type") in QUEUE_ANALYTIC_TYPES for item in expected_analytics):
                queue_status, queue = request_json(
                    f"{backend_url}/api/cameras/{camera_id}/queue",
                    token,
                )
            obstruction_status = None
            obstruction = None
            if any(item.get("type") in OBSTRUCTION_ANALYTIC_TYPES for item in expected_analytics):
                obstruction_status, obstruction = request_json(
                    f"{backend_url}/api/cameras/{camera_id}/obstruction",
                    token,
                )
            object_lifecycle_status = None
            object_lifecycle = None
            if any(item.get("type") in {"object_lifecycle", "object_removal", "object_dwell", "unattended_object_dwell"} for item in expected_analytics):
                object_lifecycle_status, object_lifecycle = request_json(
                    f"{backend_url}/api/cameras/{camera_id}/object-lifecycle",
                    token,
                )
            plate_reads_status = None
            plate_reads = None
            if any(item.get("type") in {"plate_read", "plate_read_absent", "no_plate_read"} for item in expected_analytics):
                plate_reads_status, plate_reads = request_json(
                    f"{backend_url}/api/plates/reads?cameraId={camera_id}&limit=50",
                    token,
                )
            camera = get_camera(cameras, camera_id)
            if camera:
                final_camera = camera
                max_detections = max(max_detections, int(camera.get("detectionsCount") or 0))
                for class_name, count in (camera.get("detectionClassCounts") or {}).items():
                    max_detection_class_counts[str(class_name)] = max(
                        max_detection_class_counts.get(str(class_name), 0),
                        int(count or 0),
                    )
                for class_name, count in (camera.get("recentDetectionClassCountsMax") or {}).items():
                    max_detection_class_counts[str(class_name)] = max(
                        max_detection_class_counts.get(str(class_name), 0),
                        int(count or 0),
                    )
                for sample in camera.get("recentDetectionHistory") or []:
                    max_detections = max(max_detections, int(sample.get("detectionsCount") or 0))
                final_analytics["class_counts"] = dict(max_detection_class_counts)
                final_analytics["schedule_telemetry"] = camera.get("scheduleTelemetry") or {}
            if isinstance(health, dict):
                final_health = health
            if isinstance(model_server_health, dict):
                final_model_server_health = model_server_health
                final_model_server_status = model_server_health_status
            if isinstance(alerts, list):
                final_alerts = alerts
            if isinstance(occupancy, dict):
                final_analytics["occupancy"] = occupancy
            if isinstance(queue, dict):
                final_analytics["queue"] = strongest_queue_snapshot(final_analytics.get("queue"), queue)
            if isinstance(obstruction, dict):
                final_analytics["obstruction"] = strongest_obstruction_snapshot(final_analytics.get("obstruction"), obstruction)
            if isinstance(object_lifecycle, dict):
                final_analytics["object_lifecycle"] = strongest_object_lifecycle_snapshot(
                    final_analytics.get("object_lifecycle"),
                    object_lifecycle,
                )
            if isinstance(plate_reads, list):
                final_analytics["plate_reads"] = filter_records_since(plate_reads, scenario_started_at)
            poll.update(
                {
                    "health_status": health_status,
                    "model_server_health_status": model_server_health_status,
                    "cameras_status": cameras_status,
                    "alerts_status": alerts_status,
                    "occupancy_status": occupancy_status,
                    "queue_status": queue_status,
                    "obstruction_status": obstruction_status,
                    "object_lifecycle_status": object_lifecycle_status,
                    "plate_reads_status": plate_reads_status,
                    "camera_runtime_status": camera.get("runtime_status") if camera else None,
                    "camera_status": camera.get("status") if camera else None,
                    "detections_count": camera.get("detectionsCount") if camera else None,
                    "recent_detection_class_counts_max": camera.get("recentDetectionClassCountsMax") if camera else None,
                    "alerts_count": len(alerts) if isinstance(alerts, list) else None,
                    "occupancy": summarize_occupancy(occupancy) if isinstance(occupancy, dict) else None,
                    "queue": summarize_queue(queue) if isinstance(queue, dict) else None,
                    "obstruction": summarize_obstruction(obstruction) if isinstance(obstruction, dict) else None,
                    "object_lifecycle": summarize_object_lifecycle(object_lifecycle) if isinstance(object_lifecycle, dict) else None,
                    "plate_reads": summarize_plate_reads(final_analytics.get("plate_reads", [])),
                }
            )
        except Exception as exc:
            poll["error"] = str(exc)
        result["polls"].append(poll)

        matching_alerts = filter_alerts(final_alerts, expected_rule_names, expected_policy_ids, scenario_started_at)
        analytics_ready = analytics_expectations_pass(
            {"expected_analytics": expected_analytics, "analytics_summary": summarize_analytics(final_analytics)}
        )
        alerts_ready = bool(matching_alerts) if expected_alerts else True
        if (
            final_camera
            and max_detections > 0
            and alerts_ready
            and analytics_ready
            and not expected_alerts_absent
        ):
            break
        time.sleep(poll_interval)

    stream_url = f"{backend_url}/api/stream/{camera_id}"
    stream = stream_probe(stream_url)
    matching_alerts = filter_alerts(final_alerts, expected_rule_names, expected_policy_ids, scenario_started_at)
    unexpected_alerts = (
        filter_alerts(final_alerts, forbidden_rule_names, forbidden_policy_ids, scenario_started_at)
        if expected_alerts_absent
        else []
    )
    screenshot_jpg_path = artifact_root / "screenshots" / f"{scenario_id}_live_view.jpg"
    screenshot_path = screenshot_jpg_path
    screenshot_capture = None
    if expected_ui.get("stream_should_render") and stream.get("ok"):
        screenshot_capture = capture_stream_frame(stream_url, screenshot_jpg_path)
    screenshot_exists = screenshot_path.exists()
    screenshot_stat = screenshot_path.stat() if screenshot_exists else None
    screenshot_mtime = (
        datetime.fromtimestamp(screenshot_stat.st_mtime, timezone.utc).isoformat()
        if screenshot_stat
        else None
    )
    screenshot_age_seconds = round(time.time() - screenshot_stat.st_mtime, 3) if screenshot_stat else None
    screenshot_fresh = bool(screenshot_capture and screenshot_capture.get("ok"))
    result["evidence"] = {
        "health": final_health,
        "model_server_url": model_server_url,
        "model_server_health_status": final_model_server_status,
        "model_server_health": final_model_server_health,
        "final_camera": final_camera,
        "model_preflight": model_preflight,
        "max_detections_count": max_detections,
        "max_detection_class_counts": max_detection_class_counts,
        "stream_probe": stream,
        "ui_evidence": {
            "frontend_url": scenario.get("runtime", {}).get("frontend_url"),
            "screenshot_required": bool(expected_ui.get("stream_should_render")),
            "detections_count_should_increase": expected_ui.get("detections_count_should_increase"),
            "screenshot_path": str(screenshot_path.relative_to(ROOT)),
            "screenshot_exists": screenshot_exists,
            "screenshot_fresh": screenshot_fresh,
            "screenshot_mtime": screenshot_mtime,
            "screenshot_age_seconds": screenshot_age_seconds,
            "screenshot_capture": screenshot_capture,
        },
        "expected_alerts_required": bool(expected_alerts),
        "expected_alert_absence": expected_alerts_absent,
        "expected_output_ids": sorted(expected_output_ids),
        "expected_delivery": expected_delivery,
        "webhook_capture": webhook_capture.summary() if webhook_capture else None,
        "smtp_capture": smtp_capture.summary() if smtp_capture else None,
        "expected_analytics": expected_analytics,
        "unsupported_analytics_expectation_types": unsupported_analytics_types,
        "analytics": final_analytics,
        "analytics_summary": summarize_analytics(final_analytics),
        "delivery_summary": summarize_delivery(matching_alerts),
        "matching_alerts": matching_alerts,
        "unexpected_alerts": unexpected_alerts,
        "all_recent_alerts": final_alerts[:10],
    }
    result["status"] = classify_result(result)
    result["finished_at"] = utc_now()
    write_result(result_dir, result)
    if webhook_capture:
        webhook_capture.stop()
    if smtp_capture:
        smtp_capture.stop()
    return result


def filter_alerts(alerts: list[dict[str, Any]], rule_names: set[str], policy_ids: set[str], started_at: str | None = None) -> list[dict[str, Any]]:
    matching = []
    started_ts = None
    if started_at:
        try:
            started_ts = datetime.fromisoformat(started_at).timestamp()
        except Exception:
            started_ts = None
    for alert in alerts:
        if started_ts is not None:
            try:
                alert_ts = datetime.fromisoformat(str(alert.get("timestamp"))).timestamp()
            except Exception:
                continue
            if alert_ts < started_ts:
                continue
        if rule_names and alert.get("rule") not in rule_names:
            continue
        if policy_ids and alert.get("policyId") not in policy_ids:
            continue
        matching.append(alert)
    return matching


def summarize_delivery(alerts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for alert in alerts:
        for delivery in alert.get("deliveryResults") or []:
            output_id = delivery.get("outputId") or delivery.get("id")
            if not output_id:
                continue
            entry = summary.setdefault(output_id, {"count": 0, "statuses": {}})
            entry["count"] += 1
            status = delivery.get("status") or "unknown"
            entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
    return summary


def summarize_analytics(analytics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    class_counts = analytics.get("class_counts")
    if isinstance(class_counts, dict):
        summary["class_counts"] = {
            str(class_name): int(count or 0)
            for class_name, count in class_counts.items()
        }
    occupancy = analytics.get("occupancy")
    if isinstance(occupancy, dict):
        summary["occupancy"] = summarize_occupancy(occupancy)
    queue = analytics.get("queue")
    if isinstance(queue, dict):
        summary["queue"] = summarize_queue(queue)
    obstruction = analytics.get("obstruction")
    if isinstance(obstruction, dict):
        summary["obstruction"] = summarize_obstruction(obstruction)
    object_lifecycle = analytics.get("object_lifecycle")
    if isinstance(object_lifecycle, dict):
        summary["object_lifecycle"] = summarize_object_lifecycle(object_lifecycle)
    plate_reads = analytics.get("plate_reads")
    if isinstance(plate_reads, list):
        summary["plate_reads"] = summarize_plate_reads(plate_reads)
    schedule_telemetry = analytics.get("schedule_telemetry")
    if isinstance(schedule_telemetry, dict):
        summary["schedule"] = summarize_schedule_telemetry(schedule_telemetry)
    return summary


def summarize_schedule_telemetry(schedule_telemetry: dict[str, Any]) -> dict[str, Any]:
    schedule_state = schedule_telemetry.get("scheduleState") or {}
    suppressed_capabilities = schedule_state.get("suppressedCapabilities") or []
    model_invocations = schedule_telemetry.get("modelInvocationCounts") or {}
    return {
        "ok": bool(schedule_telemetry),
        "suppressed_capabilities": [str(value) for value in suppressed_capabilities],
        "suppressed_count": int(schedule_state.get("suppressedCount") or len(suppressed_capabilities)),
        "model_invocations": {
            str(model_key): int(count or 0)
            for model_key, count in model_invocations.items()
        },
    }


def summarize_occupancy(occupancy: dict[str, Any]) -> dict[str, Any]:
    rows = occupancy.get("chairs") or occupancy.get("tables") or []
    if not isinstance(rows, list):
        rows = []
    occupied_count = sum(1 for row in rows if isinstance(row, dict) and (row.get("occupied") or row.get("status") == "occupied"))
    sample_seconds = int(occupancy.get("sampleSeconds") or occupancy.get("sessionSeconds") or 0)
    max_occupied_seconds = max(
        (int(row.get("occupiedSeconds") or 0) for row in rows if isinstance(row, dict)),
        default=int(occupancy.get("maxOccupiedSeconds") or 0),
    )
    max_longest_occupied_seconds = max(
        (int(row.get("longestOccupiedSeconds") or 0) for row in rows if isinstance(row, dict)),
        default=int(occupancy.get("maxLongestOccupiedSeconds") or 0),
    )
    duration_ready_zone_count = sum(
        1 for row in rows if isinstance(row, dict) and bool(row.get("durationReady"))
    )
    return {
        "ok": True,
        "zone_count": len(rows),
        "occupied_count": occupied_count,
        "sample_seconds": sample_seconds,
        "report_ready": bool(occupancy.get("reportReady")),
        "duration_ready": bool(occupancy.get("durationReady")),
        "duration_ready_zone_count": int(occupancy.get("durationReadyZoneCount") or duration_ready_zone_count),
        "min_duration_seconds": int(occupancy.get("minDurationSeconds") or 0),
        "max_occupied_seconds": max_occupied_seconds,
        "max_current_occupied_seconds": int(occupancy.get("maxCurrentOccupiedSeconds") or 0),
        "max_longest_occupied_seconds": max_longest_occupied_seconds,
        "total_occupied_seconds": int(occupancy.get("totalOccupiedSeconds") or 0),
        "work_hours": occupancy.get("workHours"),
    }


def summarize_queue(queue: dict[str, Any]) -> dict[str, Any]:
    zones = queue.get("queueZones") or []
    if not isinstance(zones, list):
        zones = []
    density = queue.get("maxDensityPeoplePerSquareMeter")
    density_threshold = queue.get("densityThresholdPeoplePerSquareMeter")
    return {
        "ok": True,
        "person_count": int(queue.get("personCount") or 0),
        "max_zone_count": int(queue.get("maxZoneCount") or 0),
        "zone_count": len(zones),
        "queue_active": bool(queue.get("queueActive")),
        "threshold": int(queue.get("threshold") or 0),
        "active_seconds": int(queue.get("activeSeconds") or 0),
        "current_active_seconds": int(queue.get("currentActiveSeconds") or 0),
        "longest_active_seconds": int(queue.get("longestActiveSeconds") or 0),
        "active_events": int(queue.get("activeEvents") or 0),
        "session_seconds": int(queue.get("sessionSeconds") or 0),
        "min_duration_seconds": int(queue.get("minDurationSeconds") or 0),
        "duration_ready": bool(queue.get("durationReady")),
        "calibrated": bool(queue.get("calibrated")),
        "calibrated_zone_count": int(queue.get("calibratedZoneCount") or 0),
        "max_density_people_per_square_meter": float(density) if density is not None else None,
        "density_threshold_people_per_square_meter": float(density_threshold) if density_threshold is not None else None,
        "wait_tracking_enabled": bool(queue.get("waitTimeTrackingEnabled")),
        "tracked_person_count": int(queue.get("trackedPersonCount") or 0),
        "max_wait_seconds": int(queue.get("maxWaitSeconds") or 0),
        "average_wait_seconds": float(queue.get("averageWaitSeconds") or 0),
        "oldest_wait_seconds": int(queue.get("oldestWaitSeconds") or 0),
        "wait_threshold_seconds": int(queue.get("waitThresholdSeconds") or 0),
        "wait_time_ready": bool(queue.get("waitTimeReady")),
    }


def summarize_obstruction(obstruction: dict[str, Any]) -> dict[str, Any]:
    zones = obstruction.get("routeZones") or []
    if not isinstance(zones, list):
        zones = []
    density = obstruction.get("maxObstructionDensityObjectsPerSquareMeter")
    observed_classes = set()
    for zone in zones:
        if isinstance(zone, dict):
            observed_classes.update(str(class_name) for class_name, count in (zone.get("classCounts") or {}).items() if int(count or 0) > 0)
    if not observed_classes:
        observed_classes.update(str(class_name) for class_name in obstruction.get("classes") or [])
    return {
        "ok": True,
        "object_count": int(obstruction.get("objectCount") or 0),
        "max_zone_count": int(obstruction.get("maxZoneCount") or 0),
        "zone_count": len(zones),
        "obstruction_active": bool(obstruction.get("obstructionActive")),
        "threshold": int(obstruction.get("threshold") or 0),
        "active_seconds": int(obstruction.get("activeSeconds") or 0),
        "current_active_seconds": int(obstruction.get("currentActiveSeconds") or 0),
        "longest_active_seconds": int(obstruction.get("longestActiveSeconds") or 0),
        "active_events": int(obstruction.get("activeEvents") or 0),
        "session_seconds": int(obstruction.get("sessionSeconds") or 0),
        "min_duration_seconds": int(obstruction.get("minDurationSeconds") or 0),
        "duration_ready": bool(obstruction.get("durationReady")),
        "calibrated": bool(obstruction.get("calibrated")),
        "calibrated_zone_count": int(obstruction.get("calibratedZoneCount") or 0),
        "max_density_objects_per_square_meter": float(density) if density is not None else None,
        "max_severity": str(obstruction.get("maxSeverity") or "none"),
        "max_severity_rank": int(obstruction.get("maxSeverityRank") or 0),
        "observed_classes": sorted(observed_classes),
    }


def summarize_object_lifecycle(lifecycle: dict[str, Any]) -> dict[str, Any]:
    zones = lifecycle.get("watchZones") or []
    if not isinstance(zones, list):
        zones = []
    observed_classes = set()
    seen_zone_count = 0
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        if zone.get("seenEver"):
            seen_zone_count += 1
        observed_classes.update(str(class_name) for class_name, count in (zone.get("lastClassCounts") or {}).items() if int(count or 0) > 0)
    return {
        "ok": True,
        "zone_count": len(zones),
        "seen_zone_count": seen_zone_count,
        "object_present": bool(lifecycle.get("objectPresent")),
        "dwell_detected": bool(lifecycle.get("dwellDetected")),
        "dwell_ready": bool(lifecycle.get("dwellReady")),
        "dwell_count": int(lifecycle.get("dwellCount") or 0),
        "max_present_seconds": float(lifecycle.get("maxPresentSeconds") or 0),
        "removal_detected": bool(lifecycle.get("removalDetected")),
        "removal_count": int(lifecycle.get("removalCount") or 0),
        "calibrated": bool(lifecycle.get("calibrated")),
        "observed_classes": sorted(observed_classes),
    }


def strongest_queue_snapshot(current: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(current, dict):
        return candidate
    current_score = (
        int(current.get("activeSeconds") or 0),
        int(current.get("longestActiveSeconds") or 0),
        int(current.get("maxWaitSeconds") or 0),
        int(current.get("personCount") or 0),
        int(current.get("maxZoneCount") or 0),
        1 if current.get("queueActive") else 0,
    )
    candidate_score = (
        int(candidate.get("activeSeconds") or 0),
        int(candidate.get("longestActiveSeconds") or 0),
        int(candidate.get("maxWaitSeconds") or 0),
        int(candidate.get("personCount") or 0),
        int(candidate.get("maxZoneCount") or 0),
        1 if candidate.get("queueActive") else 0,
    )
    return candidate if candidate_score > current_score else current


def strongest_obstruction_snapshot(current: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(current, dict):
        return candidate
    current_score = (
        int(current.get("activeSeconds") or 0),
        int(current.get("longestActiveSeconds") or 0),
        int(current.get("maxSeverityRank") or 0),
        float(current.get("maxObstructionDensityObjectsPerSquareMeter") or 0),
        int(current.get("objectCount") or 0),
        int(current.get("maxZoneCount") or 0),
        1 if current.get("obstructionActive") else 0,
    )
    candidate_score = (
        int(candidate.get("activeSeconds") or 0),
        int(candidate.get("longestActiveSeconds") or 0),
        int(candidate.get("maxSeverityRank") or 0),
        float(candidate.get("maxObstructionDensityObjectsPerSquareMeter") or 0),
        int(candidate.get("objectCount") or 0),
        int(candidate.get("maxZoneCount") or 0),
        1 if candidate.get("obstructionActive") else 0,
    )
    return candidate if candidate_score > current_score else current


def strongest_object_lifecycle_snapshot(current: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(current, dict):
        return candidate
    current_score = (
        int(current.get("dwellCount") or 0),
        1 if current.get("dwellDetected") else 0,
        float(current.get("maxPresentSeconds") or 0),
        int(current.get("removalCount") or 0),
        1 if current.get("removalDetected") else 0,
        1 if current.get("objectPresent") else 0,
    )
    candidate_score = (
        int(candidate.get("dwellCount") or 0),
        1 if candidate.get("dwellDetected") else 0,
        float(candidate.get("maxPresentSeconds") or 0),
        int(candidate.get("removalCount") or 0),
        1 if candidate.get("removalDetected") else 0,
        1 if candidate.get("objectPresent") else 0,
    )
    return candidate if candidate_score > current_score else current


def summarize_plate_reads(reads: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_plates = sorted({normalize_plate_text(read.get("normalizedPlate") or read.get("plateNumber")) for read in reads if read.get("normalizedPlate") or read.get("plateNumber")})
    event_types = sorted({str(read.get("eventType")) for read in reads if read.get("eventType")})
    confidences = [float(read.get("confidence")) for read in reads if isinstance(read.get("confidence"), (int, float))]
    return {
        "ok": bool(reads),
        "read_count": len(reads),
        "normalized_plates": normalized_plates,
        "event_types": event_types,
        "max_confidence": round(max(confidences), 2) if confidences else None,
        "has_snapshot": any(read.get("snapshotUrl") for read in reads),
        "has_crop": any(read.get("cropUrl") for read in reads),
    }


def normalize_plate_text(value: str | None) -> str:
    return "".join(ch for ch in (value or "").upper() if ch.isalnum())


def filter_records_since(records: list[dict[str, Any]], started_at: str) -> list[dict[str, Any]]:
    try:
        started = datetime.fromisoformat(started_at).timestamp()
    except Exception:
        return records
    filtered = []
    for record in records:
        timestamp = record.get("timestamp")
        if not timestamp:
            continue
        try:
            if datetime.fromisoformat(str(timestamp)).timestamp() >= started:
                filtered.append(record)
        except Exception:
            continue
    return filtered


def alerts_cover_outputs(alerts: list[dict[str, Any]], expected_output_ids: set[str]) -> bool:
    accepted_statuses = {"delivered", "sent", "success", "ok", "simulated"}
    delivered_output_ids = set()
    for alert in alerts:
        for delivery in alert.get("deliveryResults") or []:
            output_id = delivery.get("outputId")
            status = delivery.get("status")
            if output_id and status in accepted_statuses:
                delivered_output_ids.add(output_id)
    return expected_output_ids.issubset(delivered_output_ids)


def write_result(result_dir: Path, result: dict[str, Any]) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / f"{result['scenario_id']}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def apron_harness_readiness_gate_status(
    apron_harness_doctor: dict[str, Any] | None,
) -> str | None:
    if not apron_harness_doctor:
        return "blocked_missing_apron_harness_readiness_gate"
    if apron_harness_doctor.get("ok") is not True:
        return "blocked_invalid_apron_harness_readiness_gate"
    if not str(apron_harness_doctor.get("generated_at") or "").strip():
        return "blocked_invalid_apron_harness_readiness_gate"
    if not isinstance(apron_harness_doctor.get("pilot_gate_passed"), bool):
        return "blocked_invalid_apron_harness_readiness_gate"
    if not isinstance(apron_harness_doctor.get("production_gate_passed"), bool):
        return "blocked_invalid_apron_harness_readiness_gate"
    if not str(apron_harness_doctor.get("sales_status") or "").strip():
        return "blocked_invalid_apron_harness_readiness_gate"
    production_blockers = apron_harness_doctor.get("production_blockers") or []
    if apron_harness_doctor.get("production_gate_passed") is True and production_blockers:
        return "blocked_invalid_apron_harness_readiness_gate"
    if apron_harness_doctor.get("production_gate_passed") is False and not production_blockers:
        return "blocked_invalid_apron_harness_readiness_gate"
    return None


def apron_harness_sales_status(
    scenario_id: str,
    raw_status: str,
    apron_harness_doctor: dict[str, Any] | None,
) -> str:
    if "apron" not in scenario_id and "harness" not in scenario_id:
        return raw_status
    gate_error = apron_harness_readiness_gate_status(apron_harness_doctor)
    if gate_error:
        if raw_status and raw_status != gate_error:
            return f"{gate_error} (runtime={raw_status})"
        return gate_error
    assert apron_harness_doctor is not None
    if apron_harness_doctor.get("production_gate_passed") is True:
        return raw_status
    gate_status = str(apron_harness_doctor.get("sales_status") or "pilot_ready_not_production_compliance")
    if raw_status and raw_status != gate_status:
        return f"{gate_status} (runtime={raw_status})"
    return gate_status


def scenario_config_path(manifest: dict[str, Any], scenario: dict[str, Any]) -> Path:
    raw_config_path = (
        scenario.get("config_path")
        or (scenario.get("runtime", {}) or {}).get("config_path")
        or manifest.get("config_path", "qa/video_eval/site.yaml")
    )
    config_path = Path(str(raw_config_path))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return config_path


def _window_schedule_shape(window: dict[str, Any]) -> str:
    days = window.get("days")
    day_text = "no-days"
    if isinstance(days, list) and days:
        day_text = "/".join(str(day) for day in days)
    from_time = str(window.get("from") or "")
    to_time = str(window.get("to") or "")
    time_text = f"{from_time}-{to_time}" if from_time and to_time else "no-time"
    has_daily_weekly_shape = bool(day_text != "no-days" and time_text != "no-time")
    shape = "daily_weekly" if has_daily_weekly_shape else "incomplete"
    return f"{shape}:{day_text}:{time_text}"


def scenario_schedule_config_summary(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
) -> str:
    camera_id = str(scenario.get("camera_id") or "")
    if not camera_id:
        return "none"
    config_path = scenario_config_path(manifest, scenario)
    if not config_path.exists():
        return "missing_config"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return "unreadable_config"
    cameras = config.get("cameras") if isinstance(config.get("cameras"), dict) else {}
    camera = cameras.get(camera_id)
    if not isinstance(camera, dict):
        return "camera_not_found"
    raw_windows = camera.get("capability_windows") or camera.get("capability_active_windows")
    if not isinstance(raw_windows, dict) or not raw_windows:
        return "none"
    parts: list[str] = []
    for capability, schedule in sorted(raw_windows.items()):
        if isinstance(schedule, list):
            active = True
            windows = schedule
        elif isinstance(schedule, dict):
            active = schedule.get("active", True) is not False
            windows = schedule.get("windows") or []
        else:
            continue
        if not isinstance(windows, list) or not windows:
            parts.append(f"{capability}:{'active' if active else 'inactive'}:no-windows")
            continue
        shapes = [
            _window_schedule_shape(window)
            for window in windows
            if isinstance(window, dict)
        ]
        shape_text = "+".join(shapes) if shapes else "no-valid-windows"
        parts.append(f"{capability}:{'active' if active else 'inactive'}:{shape_text}")
    return "capability_windows(" + ", ".join(parts) + ")" if parts else "none"


def report(manifest_path: Path) -> tuple[Path, Path]:
    manifest = load_manifest(manifest_path)
    artifact_root = ROOT / manifest.get("artifact_root", "qa/video_eval")
    result_dir = artifact_root / "results"
    sales_path = artifact_root / "SALES_READINESS_REPORT.md"
    claims_path = artifact_root / "CLAIMS_MATRIX.md"
    coverage = manifest.get("coverage_boundaries") or {}
    skipped_model_packs = {
        str(value)
        for value in coverage.get("skipped_model_packs") or []
        if str(value).strip()
    }
    device_probe_path = result_dir / "model_pack_device_probe.json"
    evidence_doctor_path = result_dir / "model_pack_evidence_doctor.json"
    apron_harness_doctor_path = result_dir / "apron_harness_readiness_doctor.json"
    device_probe = filter_model_pack_scope(load_model_pack_device_probe(device_probe_path), skipped_model_packs)
    evidence_doctor = filter_model_pack_scope(load_model_pack_evidence_doctor(evidence_doctor_path), skipped_model_packs)
    apron_harness_doctor = load_apron_harness_readiness_doctor(apron_harness_doctor_path)
    skipped_scenario_ids = {
        str(value)
        for value in coverage.get("skipped_scenario_ids") or []
        if str(value).strip()
    }
    skipped_verticals = {
        str(value)
        for value in coverage.get("skipped_verticals") or []
        if str(value).strip()
    }
    scenario_by_id = {scenario.get("id"): scenario for scenario in manifest.get("scenarios", [])}
    results = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scenario_id = str(result.get("scenario_id") or "")
        if not scenario_id or scenario_id in skipped_scenario_ids:
            continue
        scenario = scenario_by_id.get(scenario_id) or {}
        if str(scenario.get("vertical") or "") in skipped_verticals:
            continue
        results.append(result)
    result_by_id = {result.get("scenario_id"): result for result in results}

    sales_lines = [
        "# Rakshak Lens Video Analytics Sales-Readiness Report",
        "",
        f"Updated: {utc_now()}",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Sales Status | Runtime Status | Video | Evidence Summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        scenario_id = str(result.get("scenario_id") or "")
        scenario = scenario_by_id.get(scenario_id) or {}
        raw_status = str(result.get("status") or "not_run")
        sales_status = apron_harness_sales_status(scenario_id, raw_status, apron_harness_doctor)
        evidence = result.get("evidence", {})
        camera = evidence.get("final_camera") or {}
        delivery_summary = format_delivery_evidence(evidence)
        analytics_summary = format_analytics_summary(evidence.get("analytics_summary") or {})
        schedule_config_summary = scenario_schedule_config_summary(manifest, scenario)
        ui = evidence.get("ui_evidence") or {}
        unexpected_alerts = len(evidence.get("unexpected_alerts") or [])
        negative_alert_summary = (
            f", forbidden alerts={unexpected_alerts}"
            if evidence.get("expected_alert_absence")
            else ""
        )
        summary = (
            f"source={result.get('source', {}).get('type', 'unknown')}, "
            f"stream_type={camera.get('stream_type', 'unknown')}, "
            f"max detections={evidence.get('max_detections_count', 0)}, "
            f"matching alerts={len(evidence.get('matching_alerts') or [])}{negative_alert_summary}, "
            f"stream={bool((evidence.get('stream_probe') or {}).get('ok'))}, "
            f"analytics={analytics_summary}, "
            f"schedule_config={schedule_config_summary}, "
            f"delivery={delivery_summary}, "
            f"ui_screenshot={bool(ui.get('screenshot_exists'))}, fresh={bool(ui.get('screenshot_fresh'))}"
        )
        sales_lines.append(
            f"| `{scenario_id}` | `{sales_status}` | `{raw_status}` | `{result.get('video')}` | {summary} |"
        )
    if not results:
        sales_lines.append("| _none_ | `not_run` | `not_run` |  | No scenario results yet. |")

    device_gate_lines = format_model_pack_device_gate(device_probe, apron_harness_doctor)
    if device_gate_lines:
        sales_lines.extend(["", "## Model Pack Device Gate", "", *device_gate_lines])
    evidence_gate_lines = format_model_pack_evidence_gate(evidence_doctor, apron_harness_doctor)
    if evidence_gate_lines:
        sales_lines.extend(["", "## Model Pack Evidence Gate", "", *evidence_gate_lines])
    apron_harness_gate_lines = format_apron_harness_readiness_gate(apron_harness_doctor)
    if apron_harness_gate_lines:
        sales_lines.extend(["", "## Apron/Harness Production Gate", "", *apron_harness_gate_lines])

    if coverage:
        tested_brochures = coverage.get("tested_final_send_brochures") or []
        if tested_brochures:
            sales_lines.extend(
                [
                    "",
                    "## Brochure Coverage",
                    "",
                    "| Brochure | Final-send File | Scenario IDs | Current Evidence Status |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for item in tested_brochures:
                scenario_ids = [str(value) for value in item.get("scenario_ids") or []]
                statuses = []
                for scenario_id in scenario_ids:
                    raw_status = str(result_by_id.get(scenario_id, {}).get("status", "not_run"))
                    status = apron_harness_sales_status(scenario_id, raw_status, apron_harness_doctor)
                    statuses.append(f"{scenario_id}: {status}")
                sales_lines.append(
                    "| "
                    f"`{item.get('code', 'unknown')}` | "
                    f"`{item.get('file', '')}` | "
                    f"{format_code_list(scenario_ids)} | "
                    f"{'; '.join(statuses) or 'not_run'} |"
                )
        skipped_brochures = coverage.get("skipped_final_send_brochures") or []
        if skipped_brochures:
            sales_lines.extend(["", "## Skipped Brochures", ""])
            for item in skipped_brochures:
                sales_lines.append(
                    f"- `{item.get('code', 'unknown')}`: {item.get('reason', 'Skipped for current validation scope.')}"
                )
        caveats = coverage.get("evidence_caveats") or []
        if caveats:
            sales_lines.extend(["", "## Evidence Caveats", ""])
            sales_lines.extend(f"- {caveat}" for caveat in caveats)
        unproven = coverage.get("not_yet_proven_claims") or []
        if unproven:
            sales_lines.extend(
                [
                    "",
                    "## Not Yet Proven From This Run",
                    "",
                    "| Claim Boundary | Affected Brochures | Status | Note |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for item in unproven:
                sales_lines.append(
                    "| "
                    f"{item.get('claim', '')} | "
                    f"{format_code_list(item.get('affected_brochures') or [])} | "
                    f"`{item.get('status', 'not_tested')}` | "
                    f"{item.get('note', '')} |"
                )

    claims_lines = [
        "# Rakshak Lens Video Analytics Claims Matrix",
        "",
        f"Updated: {utc_now()}",
        "",
        "| Claim | Scenario | Status | Evidence | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scenario_id, scenario in scenario_by_id.items():
        if str(scenario_id or "") in skipped_scenario_ids:
            continue
        if str(scenario.get("vertical") or "") in skipped_verticals:
            continue
        result = result_by_id.get(scenario_id, {})
        raw_status = str(result.get("status", "not_run"))
        status = apron_harness_sales_status(str(scenario_id), raw_status, apron_harness_doctor)
        evidence = result.get("evidence", {})
        camera = evidence.get("final_camera") or {}
        delivery_summary = format_delivery_evidence(evidence)
        analytics_summary = format_analytics_summary(evidence.get("analytics_summary") or {})
        schedule_config_summary = scenario_schedule_config_summary(manifest, scenario)
        ui = evidence.get("ui_evidence") or {}
        unexpected_alerts = len(evidence.get("unexpected_alerts") or [])
        negative_alert_summary = (
            f", forbidden_alerts={unexpected_alerts}"
            if evidence.get("expected_alert_absence")
            else ""
        )
        evidence_summary = (
            f"source={scenario.get('source', {}).get('type', 'unknown')}, "
            f"stream_type={camera.get('stream_type', 'unknown')}, "
            f"detections={evidence.get('max_detections_count', 0)}, "
            f"alerts={len(evidence.get('matching_alerts') or [])}{negative_alert_summary}, "
            f"analytics={analytics_summary}, "
            f"schedule_config={schedule_config_summary}, "
            f"delivery={delivery_summary}, "
            f"ui_screenshot={bool(ui.get('screenshot_exists'))}, fresh={bool(ui.get('screenshot_fresh'))}"
        )
        for claim in scenario.get("sales_claims", []):
            claims_lines.append(
                f"| {claim} | `{scenario_id}` | `{status}` | {evidence_summary} | Source: {scenario.get('source', {}).get('provider', 'unknown')} |"
            )
    if device_gate_lines:
        claims_lines.extend(["", "## Model Pack Device Gate", "", *device_gate_lines])
    if evidence_gate_lines:
        claims_lines.extend(["", "## Model Pack Evidence Gate", "", *evidence_gate_lines])
    if apron_harness_gate_lines:
        claims_lines.extend(["", "## Apron/Harness Production Gate", "", *apron_harness_gate_lines])
    unproven = coverage.get("not_yet_proven_claims") or []
    if unproven:
        claims_lines.extend(
            [
                "",
                "## Not Yet Proven / Do Not Claim From Current Evidence",
                "",
                "| Claim Boundary | Affected Brochures | Status | Required Evidence |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in unproven:
            claims_lines.append(
                "| "
                f"{item.get('claim', '')} | "
                f"{format_code_list(item.get('affected_brochures') or [])} | "
                f"`{item.get('status', 'not_tested')}` | "
                f"{item.get('note', '')} |"
            )

    sales_path.write_text("\n".join(sales_lines) + "\n", encoding="utf-8")
    claims_path.write_text("\n".join(claims_lines) + "\n", encoding="utf-8")
    return sales_path, claims_path


def filter_model_pack_scope(report: dict[str, Any] | None, skipped_model_packs: set[str]) -> dict[str, Any] | None:
    if not report or not skipped_model_packs:
        return report
    filtered = dict(report)
    packs = report.get("packs")
    if isinstance(packs, dict):
        filtered["packs"] = {
            key: value
            for key, value in packs.items()
            if str(key) not in skipped_model_packs
        }
        stats = dict(report.get("stats") or {})
        if stats:
            stats["pack_count"] = len(filtered["packs"])
            filtered["stats"] = stats
    filtered["skipped_model_packs"] = sorted(skipped_model_packs)
    return filtered


def load_model_pack_device_probe(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def load_model_pack_evidence_doctor(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def load_apron_harness_readiness_doctor(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def format_model_pack_device_gate(
    probe: dict[str, Any] | None,
    apron_harness_doctor: dict[str, Any] | None = None,
) -> list[str]:
    if not probe:
        return []
    torch_info = probe.get("torch") or {}
    platform_info = probe.get("platform") or {}
    macos_info = platform_info.get("macos") if isinstance(platform_info.get("macos"), dict) else {}
    packs = probe.get("packs") or {}
    mps_probe_ok = torch_info.get("mps_probe_ok")
    mps_probe_ok_text = "unknown" if mps_probe_ok is None else str(bool(mps_probe_ok))
    macos_version = macos_info.get("product_version") or platform_info.get("mac_ver")
    macos_build = macos_info.get("build_version")
    runtime_parts = [
        f"{platform_info.get('system', 'unknown')} {platform_info.get('machine', 'unknown')}",
        f"Python `{platform_info.get('python', 'unknown')}`",
    ]
    if macos_info or platform_info.get("system") == "Darwin":
        macos_text = f"macOS `{macos_version or 'unknown'}`"
        if macos_build:
            macos_text += f" build `{macos_build}`"
        runtime_parts.append(macos_text)
    lines = [
        f"- Probe: `qa/video_eval/results/model_pack_device_probe.json` generated `{probe.get('generated_at', 'unknown')}` on `{probe.get('host', 'unknown')}`.",
        (
            "- Local runtime: "
            f"{', '.join(runtime_parts)}, "
            f"torch `{torch_info.get('version', 'unknown')}`, "
            f"MPS built=`{bool(torch_info.get('mps_built'))}`, "
            f"MPS available=`{bool(torch_info.get('mps_available'))}`, "
            f"MPS probe ok=`{mps_probe_ok_text}`, "
            f"CUDA available=`{bool(torch_info.get('cuda_available'))}`."
        ),
        f"- Acceptance gate: `{probe.get('mps_acceptance_gate', 'unknown')}`.",
        f"- Local performance gate: `{probe.get('local_performance_acceptance_gate', 'unknown')}`.",
    ]
    artifact_layout = probe.get("artifact_layout") or {}
    if artifact_layout:
        unexpected_paths = [str(path) for path in artifact_layout.get("unexpected_paths") or []]
        lines.append(
            "- Model artifact layout gate: "
            f"`{artifact_layout.get('gate', probe.get('model_artifact_layout_gate', 'unknown'))}`, "
            f"unexpected legacy artifacts=`{len(unexpected_paths)}`."
        )
        if unexpected_paths:
            rendered_paths = ", ".join(f"`{path}`" for path in unexpected_paths[:5])
            if len(unexpected_paths) > 5:
                rendered_paths += f", ... +{len(unexpected_paths) - 5} more"
            lines.append(f"- Unexpected legacy model artifacts: {rendered_paths}.")
    elif probe.get("model_artifact_layout_gate"):
        lines.append(f"- Model artifact layout gate: `{probe.get('model_artifact_layout_gate')}`.")
    if torch_info.get("mps_runtime_error"):
        runtime_error = str(torch_info.get("mps_runtime_error")).replace("`", "'")
        lines.append(f"- MPS runtime error: `{runtime_error}`.")
    if packs:
        pack_parts = []
        for pack_id in sorted(packs):
            reasons = "/".join(str(reason) for reason in packs[pack_id].get("local_device_reasons", [])) or "unknown"
            functional = bool(packs[pack_id].get("local_functional_satisfied", packs[pack_id].get("local_device_satisfied")))
            performance = bool(packs[pack_id].get("local_performance_gate_satisfied"))
            status = "perf-ok" if performance else ("wiring-only" if functional else "blocked")
            pack_parts.append(f"{pack_id}:{status}({reasons})")
        lines.append(f"- Pack local-device/performance status: {', '.join(pack_parts)}.")
    if (
        isinstance(apron_harness_doctor, dict)
        and "factory_ppe_3cam" in packs
        and apron_harness_doctor.get("production_gate_passed") is not True
    ):
        factory_ppe = packs.get("factory_ppe_3cam") or {}
        performance = bool(factory_ppe.get("local_performance_gate_satisfied"))
        functional = bool(factory_ppe.get("local_functional_satisfied", factory_ppe.get("local_device_satisfied")))
        factory_ppe_status = "perf-ok" if performance else ("wiring-only" if functional else "blocked")
        lines.append(
            "- Factory PPE device qualifier: "
            f"factory_ppe_3cam local MPS/device status is `{factory_ppe_status}`, "
            f"but apron/harness production sales status is "
            f"`{apron_harness_doctor.get('sales_status', 'pilot_ready_not_production_compliance')}` "
            "until closed-set data, model registry, promotion, and Jetson gates pass."
        )
    skipped_model_packs = [str(pack_id) for pack_id in probe.get("skipped_model_packs") or []]
    if skipped_model_packs:
        lines.append(f"- Skipped model packs for current sales scope: {format_code_list(skipped_model_packs)}.")
    if probe.get("mps_acceptance_gate") == "mps_unavailable_cpu_fallback_only":
        lines.append("- Current local evidence is CPU fallback evidence; it proves model-pack wiring, not Apple Silicon MPS performance.")
    return lines


def format_model_pack_evidence_gate(
    report: dict[str, Any] | None,
    apron_harness_doctor: dict[str, Any] | None = None,
) -> list[str]:
    if not report:
        return []
    stats = report.get("stats") or {}
    packs = report.get("packs") or {}
    lines = [
        f"- Probe: `qa/video_eval/results/model_pack_evidence_doctor.json` generated `{report.get('generated_at', 'unknown')}`.",
        f"- Acceptance gate: `{'pass' if report.get('ok') else 'failed'}`.",
        (
            "- Coverage: "
            f"packs=`{stats.get('pack_count', 0)}`, "
            f"scenarios=`{stats.get('unique_scenario_count', 0)}`, "
            f"focused_configs=`{stats.get('unique_config_count', 0)}`, "
            f"ready_results=`{stats.get('ready_result_count', 0)}`, "
            f"yaml_apply_skipped=`{stats.get('yaml_apply_skipped_count', 0)}`, "
            f"log_checks=`{stats.get('log_evidence_check_count', 0)}`, "
            f"active_window_checks=`{stats.get('active_window_check_count', 0)}`, "
            f"capability_window_config_checks=`{stats.get('capability_window_config_check_count', 0)}`, "
            f"delivery_checks=`{stats.get('delivery_check_count', 0)}`, "
            f"detector_window_checks=`{stats.get('detector_suppression_check_count', 0)}`."
        ),
    ]
    if packs:
        pack_parts = []
        for pack_id in sorted(packs):
            pack = packs[pack_id] or {}
            status = "ok" if pack.get("ok") else "failed"
            pack_parts.append(f"{pack_id}:{status}({pack.get('scenario_count', 0)} scenarios)")
        lines.append(f"- Pack evidence status: {', '.join(pack_parts)}.")
    if (
        isinstance(apron_harness_doctor, dict)
        and "factory_ppe_3cam" in packs
        and apron_harness_doctor.get("production_gate_passed") is not True
    ):
        factory_ppe = packs.get("factory_ppe_3cam") or {}
        factory_ppe_status = "ok" if factory_ppe.get("ok") else "failed"
        lines.append(
            "- Factory PPE production qualifier: "
            f"factory_ppe_3cam local evidence is `{factory_ppe_status}`, "
            f"but apron/harness production sales status is "
            f"`{apron_harness_doctor.get('sales_status', 'pilot_ready_not_production_compliance')}` "
            "until closed-set data, model registry, promotion, and Jetson gates pass."
        )
    skipped_model_packs = [str(pack_id) for pack_id in report.get("skipped_model_packs") or []]
    if skipped_model_packs:
        lines.append(f"- Skipped model packs for current sales scope: {format_code_list(skipped_model_packs)}.")
    errors = [str(error) for error in report.get("errors") or []]
    warnings = [str(warning) for warning in report.get("warnings") or []]
    if errors:
        lines.append(f"- Errors: {'; '.join(errors[:5])}{'; ...' if len(errors) > 5 else ''}")
    if warnings:
        lines.append(f"- Warnings: {'; '.join(warnings[:5])}{'; ...' if len(warnings) > 5 else ''}")
    return lines


def format_apron_harness_readiness_gate(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return []
    capabilities = report.get("capabilities") or {}
    lines = [
        f"- Probe: `qa/video_eval/results/apron_harness_readiness_doctor.json` generated `{report.get('generated_at', 'unknown')}`.",
        f"- Pilot gate: `{'pass' if report.get('pilot_gate_passed') else 'failed'}`.",
        f"- Production gate: `{'pass' if report.get('production_gate_passed') else 'blocked'}`.",
        f"- Sales status: `{report.get('sales_status', 'unknown')}`.",
        f"- Production blocker count: `{report.get('production_blocker_count', len(report.get('production_blockers') or []))}`.",
        f"- Source status: `{report.get('sourcing_status', 'unknown')}` "
        f"(seed candidates=`{report.get('sourcing_candidate_count', 0)}`).",
    ]
    optional_status = report.get("optional_gate_status") if isinstance(report.get("optional_gate_status"), dict) else {}
    if optional_status.get("seed_source_review") or report.get("seed_source_review"):
        lines.append(
            "- Seed-source review: "
            f"`{optional_status.get('seed_source_review', 'unknown')}`, "
            f"training_usable=`{report.get('seed_source_review_training_usable_count', 0)}`."
        )
    if optional_status.get("seed_source_review_bundle") or report.get("seed_source_review_bundle"):
        bundle = report.get("seed_source_review_bundle") if isinstance(report.get("seed_source_review_bundle"), dict) else {}
        lines.append(
            "- Source-review bundle: "
            f"`{optional_status.get('seed_source_review_bundle', 'unknown')}`, "
            f"hashes_ok=`{report.get('seed_source_review_bundle_ok')}`, "
            f"artifacts=`{bundle.get('checked_artifact_count', bundle.get('artifact_count', report.get('seed_source_review_bundle_artifact_count', 0)))}`."
        )
    seed_source_next_reviews = report.get("seed_source_next_review_queue")
    if isinstance(seed_source_next_reviews, list) and seed_source_next_reviews:
        parts = []
        for item in seed_source_next_reviews[:3]:
            if not isinstance(item, dict):
                continue
            priority = item.get("review_priority", "unknown")
            capability = item.get("capability", "unknown")
            source_ref = item.get("source_ref", "unknown")
            source_url = item.get("source_url") or item.get("url") or ""
            license_note = item.get("license_note") or "license_unrecorded"
            packet = item.get("review_packet_path") or "missing_packet"
            details = f"priority={priority}, license={license_note}, packet=`{packet}`"
            if source_url:
                details += f", url={source_url}"
            parts.append(f"{capability}/{source_ref}({details})")
        if parts:
            lines.append(f"- Next seed-source reviews: {', '.join(parts)}.")
        fill_parts = []
        for item in seed_source_next_reviews[:3]:
            if not isinstance(item, dict):
                continue
            fill_plan = (
                item.get("seed_import_fill_plan")
                if isinstance(item.get("seed_import_fill_plan"), dict)
                else {}
            )
            if not fill_plan:
                continue
            source_ref = item.get("source_ref", "unknown")
            required_classes = ", ".join(
                str(value)
                for value in fill_plan.get("required_local_classes") or []
                if str(value)
            )
            missing = ", ".join(
                str(value)
                for value in fill_plan.get("missing_required_classes_from_suggestion") or []
                if str(value)
            ) or "none"
            nonzero = ", ".join(
                str(value)
                for value in fill_plan.get("expected_count_classes_that_must_be_nonzero") or []
                if str(value)
            )
            fill_parts.append(
                f"{source_ref}(classes={required_classes or 'unknown'}, "
                f"missing_suggestion={missing}, nonzero_counts={nonzero or 'unknown'})"
            )
        if fill_parts:
            lines.append(f"- Seed import fill plans: {', '.join(fill_parts)}.")
    seed_source_minimum_approval_path = report.get("seed_source_minimum_approval_path")
    if isinstance(seed_source_minimum_approval_path, dict) and seed_source_minimum_approval_path.get("checked"):
        boundary = str(seed_source_minimum_approval_path.get("evidence_boundary", "not approval")).rstrip(".")
        source_refs = ", ".join(
            str(value)
            for value in seed_source_minimum_approval_path.get("minimum_review_source_refs") or []
            if str(value)
        )
        capability_parts = []
        minimum_path_capabilities = seed_source_minimum_approval_path.get("capabilities")
        if isinstance(minimum_path_capabilities, dict):
            for capability in ("apron_required", "harness_required"):
                item = minimum_path_capabilities.get(capability)
                if not isinstance(item, dict):
                    continue
                selected_sources = [
                    str(source.get("source_ref"))
                    for source in item.get("selected_sources") or []
                    if isinstance(source, dict) and source.get("source_ref")
                ]
                if selected_sources:
                    capability_parts.append(f"{capability}={'+'.join(selected_sources)}")
        lines.append(
            "- Minimum seed-source approval path: "
            f"sources=`{source_refs or 'none'}`, "
            f"capabilities=`{', '.join(capability_parts) or 'none'}`, "
            f"coverage_gaps=`{seed_source_minimum_approval_path.get('coverage_gap_count')}`, "
            f"training_usable=`{seed_source_minimum_approval_path.get('training_usable_count', 0)}`, "
            f"boundary={boundary}."
        )
    if optional_status.get("seed_import_manifest") or report.get("seed_import_manifest_review"):
        lines.append(
            "- Seed-import manifest: "
            f"`{optional_status.get('seed_import_manifest', 'unknown')}`, "
            f"included=`{report.get('seed_import_manifest_included_count', 0)}`, "
            f"approved=`{report.get('seed_import_manifest_approved_count', 0)}`."
        )
    minimum_seed_import_template = report.get("minimum_seed_import_manifest_template_summary")
    if isinstance(minimum_seed_import_template, dict) and minimum_seed_import_template.get("available"):
        consistency = report.get("minimum_seed_import_manifest_template_consistency")
        consistency_text = ""
        if isinstance(consistency, dict) and consistency.get("checked"):
            consistency_text = (
                f", consistency_valid=`{consistency.get('valid')}`, "
                f"refs_match=`{consistency.get('source_refs_match')}`"
            )
        lines.append(
            "- Minimum seed-import template: "
            f"path=`{minimum_seed_import_template.get('path')}`, "
            f"scope=`{minimum_seed_import_template.get('template_scope')}`, "
            f"sources=`{', '.join(str(value) for value in minimum_seed_import_template.get('selected_source_refs') or [])}`, "
            f"imports=`{minimum_seed_import_template.get('import_count', 0)}`, "
            f"enabled=`{minimum_seed_import_template.get('enabled_import_count', 0)}`"
            f"{consistency_text}."
        )
    seed_import_fill_contract = report.get("seed_import_fill_contract_summary")
    if isinstance(seed_import_fill_contract, dict) and seed_import_fill_contract.get("available"):
        forbidden = seed_import_fill_contract.get("forbidden_until_approved") or []
        commands = seed_import_fill_contract.get("validation_commands") or []
        lines.append(
            "- Seed import fill contract: "
            f"required_before_include=`{seed_import_fill_contract.get('required_before_include_in_training_count', 0)}`, "
            f"forbidden_until_approved={format_code_list(forbidden)}, "
            f"validation_commands=`{len(commands)}`."
        )
    seed_import_export_preflight = report.get("seed_import_export_preflight_summary")
    if isinstance(seed_import_export_preflight, dict) and seed_import_export_preflight:
        checks = [
            str(value)
            for value in seed_import_export_preflight.get("checks") or []
            if str(value)
        ]
        lines.append(
            "- Seed export preflight: "
            f"field=`{seed_import_export_preflight.get('required_manifest_field', 'raw_export_local_path')}`, "
            f"blocked_reason=`{seed_import_export_preflight.get('blocked_reason', 'unknown')}`, "
            f"checked=`{seed_import_export_preflight.get('preflight_checked_count', 0)}`, "
            f"approved=`{seed_import_export_preflight.get('preflight_approved_count', 0)}`, "
            f"missing_local_zip=`{seed_import_export_preflight.get('missing_raw_export_local_path_count', 0)}`, "
            f"review_artifacts_checked=`{seed_import_export_preflight.get('review_artifact_checked_count', 0)}`, "
            f"review_artifact_errors=`{seed_import_export_preflight.get('review_artifact_error_count', 0)}`."
        )
        top_blockers = [
            str(value)
            for value in seed_import_export_preflight.get("top_blockers") or []
            if str(value)
        ]
        if top_blockers:
            lines.append(
                "- Seed export preflight blockers: "
                f"{'; '.join(top_blockers)}"
                f"{'; ...' if seed_import_export_preflight.get('blocker_count', 0) > len(top_blockers) else ''}."
            )
        if checks:
            lines.append(f"- Seed export proof checks: {format_code_list(checks)}.")
        lines.append(
            "- Seed export materialization gate: command exit `0` is not enough; "
            "the emitted `.seed_export_import.json` sidecar must validate after materialization."
        )
    if optional_status.get("model_registry"):
        lines.append(f"- Model registry report: `{optional_status.get('model_registry')}`.")
    model_registry_handoff = report.get("model_registry_handoff")
    if isinstance(model_registry_handoff, dict) and model_registry_handoff.get("checked"):
        lines.append(
            "- Model registry handoff: "
            f"status=`{model_registry_handoff.get('status')}`, "
            f"registry_status=`{model_registry_handoff.get('registry_status')}`, "
            f"model_definition_valid=`{model_registry_handoff.get('model_definition_valid')}`, "
            f"destination_exists=`{model_registry_handoff.get('destination_exists')}`, "
            f"metadata_valid=`{model_registry_handoff.get('metadata_valid')}`."
        )
    jetson_template_handoff = report.get("jetson_template_handoff")
    if isinstance(jetson_template_handoff, dict) and jetson_template_handoff.get("checked"):
        lines.append(
            "- Jetson template handoff: "
            f"status=`{jetson_template_handoff.get('status')}`, "
            f"valid_templates=`{jetson_template_handoff.get('valid_template_contract_count', 0)}/"
            f"{jetson_template_handoff.get('template_count', 0)}`, "
            f"identity_stamped=`{jetson_template_handoff.get('identity_stamped_count', 0)}/"
            f"{jetson_template_handoff.get('template_count', 0)}`."
        )
    model_definition = report.get("closed_set_model_manager_definition")
    if isinstance(model_definition, dict):
        lines.append(
            "- Closed-set model definition: "
            f"registered=`{model_definition.get('registered')}`, "
            f"valid=`{model_definition.get('valid')}`, "
            f"artifact_exists=`{model_definition.get('artifact_exists')}`, "
            f"registry_path=`{model_definition.get('registry_path', 'unknown')}`."
        )
    candidate_templates = report.get("closed_set_candidate_yaml_templates")
    if isinstance(candidate_templates, dict):
        template_rows = candidate_templates.get("templates") if isinstance(candidate_templates.get("templates"), list) else []
        required_model_plan_ok = sum(
            1
            for row in template_rows
            if isinstance(row, dict) and row.get("required_model_plan_ok") is True
        )
        cli_preflight_ok = sum(
            1
            for row in template_rows
            if isinstance(row, dict) and row.get("cli_preflight_ok") is True
        )
        one_at_a_time_ok = sum(
            1
            for row in template_rows
            if isinstance(row, dict) and row.get("one_at_a_time_ok") is True
        )
        lines.append(
            "- Closed-set candidate YAML preflight: "
            f"valid=`{candidate_templates.get('valid')}`, "
            f"templates=`{candidate_templates.get('valid_template_count', 0)}/"
            f"{candidate_templates.get('template_count', 0)}`, "
            f"cli_validate_plan=`{cli_preflight_ok}/{len(template_rows)}`, "
            f"required_model_plan=`{required_model_plan_ok}/{len(template_rows)}`, "
            f"one_at_a_time=`{one_at_a_time_ok}/{len(template_rows)}`."
        )
        template_blockers = [
            str(blocker)
            for blocker in candidate_templates.get("blockers") or []
            if str(blocker)
        ]
        if template_blockers:
            lines.append(
                "- Closed-set candidate YAML blockers: "
                f"{'; '.join(template_blockers[:3])}{'; ...' if len(template_blockers) > 3 else ''}."
            )
    candidate_runtime = report.get("closed_set_candidate_runtime_evidence")
    if isinstance(candidate_runtime, dict):
        lines.append(
            "- Closed-set candidate runtime evidence: "
            f"valid=`{candidate_runtime.get('valid')}`, "
            f"files_present=`{candidate_runtime.get('present_result_count', 0)}/{candidate_runtime.get('result_count', 0)}`, "
            f"valid_results=`{candidate_runtime.get('valid_result_count', 0)}/{candidate_runtime.get('result_count', 0)}`, "
            f"missing_model_preflight_blocks=`{candidate_runtime.get('preflight_blocked_missing_model_count', 0)}`."
        )
        result_rows = candidate_runtime.get("results") if isinstance(candidate_runtime.get("results"), list) else []
        blockers = []
        for row in result_rows:
            if not isinstance(row, dict):
                continue
            errors = row.get("errors") if isinstance(row.get("errors"), list) else []
            if errors:
                blockers.append(f"{row.get('scenario_id', 'unknown')}:{errors[0]}")
            if len(blockers) >= 3:
                break
        if blockers:
            lines.append(f"- Closed-set candidate runtime blockers: {'; '.join(blockers)}.")
    if capabilities:
        parts = []
        for capability in sorted(capabilities):
            item = capabilities[capability] or {}
            guard = ((item.get("scenarios") or {}).get("false_positive_guard") or {})
            suppression = ((item.get("scenarios") or {}).get("suppression") or {})
            invocations = suppression.get("model_invocations") or {}
            parts.append(
                f"{capability}:{'ok' if item.get('ok') else 'failed'}"
                f"(visible={guard.get('visible_class_total', 0)}, "
                f"ppe_specialist_off_invocations={invocations.get('ppe_specialist', 'unknown')})"
            )
        lines.append(f"- Capability evidence: {', '.join(parts)}.")
    handoff = report.get("closed_set_handoff") or {}
    if handoff:
        missing = handoff.get("missing_label_minimums") or {}
        production_missing = handoff.get("production_missing_label_minimums") or {}
        required_counts = handoff.get("required_labeled_images_per_class") or {}
        capture_deficit = handoff.get("capture_deficit") or {}
        production_capture_deficit = handoff.get("production_capture_deficit") or {}
        lines.append(
            "- Closed-set handoff: "
            f"dataset_schema={'pass' if handoff.get('dataset_schema_ok') else 'failed'}, "
            f"training_ready=`{_closed_set_training_ready_status(handoff)}`, "
            f"dry_run=`{handoff.get('training_dry_run_status', 'unknown')}`, "
            f"model=`{handoff.get('training_model', 'unknown')}`, "
            f"device=`{handoff.get('selected_device', 'unknown')}`, "
            f"missing_label_classes=`{len(missing)}`, "
            f"production_missing_label_classes=`{len(production_missing)}`, "
            f"training_preflight=`{_training_preflight_status(handoff)}`."
        )
        training_torch = (
            handoff.get("training_torch_status")
            if isinstance(handoff.get("training_torch_status"), dict)
            else {}
        )
        if training_torch:
            lines.append(
                "- Local training device gate: "
                f"selected=`{handoff.get('selected_device', 'unknown')}`, "
                f"torch=`{training_torch.get('version', 'unknown')}`, "
                f"mps_built=`{training_torch.get('mps_built')}`, "
                f"mps_available=`{training_torch.get('mps_available')}`, "
                f"mps_probe_ok=`{training_torch.get('mps_probe_ok')}`, "
                f"cuda_available=`{training_torch.get('cuda_available')}`."
            )
        production_preflight = (
            handoff.get("production_training_plan_preflight")
            if isinstance(handoff.get("production_training_plan_preflight"), dict)
            else {}
        )
        if production_preflight:
            inputs = production_preflight.get("inputs") if isinstance(production_preflight.get("inputs"), dict) else {}
            input_parts = [
                f"{key}={value}"
                for key, value in sorted(inputs.items())
                if str(value)
            ]
            lines.append(
                "- Production training preflight: "
                f"checked=`{production_preflight.get('checked')}`, "
                f"ok=`{production_preflight.get('ok')}`, "
                f"error=`{production_preflight.get('error') or 'none'}`, "
                f"inputs={format_code_list(input_parts)}."
            )
        if required_counts:
            lines.append(
                "- Closed-set label minimums: "
                f"pilot=`{required_counts.get('pilot', 'unknown')}` per class, "
                f"production=`{required_counts.get('production', 'unknown')}` per class."
            )
        if capture_deficit:
            batches = capture_deficit.get("next_capture_batches") or []
            matrix_rows = sum(
                len(batch.get("capture_matrix") or [])
                for batch in batches
                if isinstance(batch, dict)
            )
            lines.append(
                "- Capture deficit: "
                f"missing_label_annotations=`{capture_deficit.get('total_missing_label_annotations', 0)}`, "
                f"recommended_label_review_rows=`{capture_deficit.get('recommended_label_review_rows', 0)}`, "
                f"coverage_deficits=`{capture_deficit.get('coverage_deficit_count', 0)}`, "
                f"next_batches=`{len(batches)}`, "
                f"matrix_rows=`{matrix_rows}`."
            )
        if production_capture_deficit:
            production_batches = production_capture_deficit.get("next_capture_batches") or []
            production_matrix_rows = sum(
                len(batch.get("capture_matrix") or [])
                for batch in production_batches
                if isinstance(batch, dict)
            )
            lines.append(
                "- Production capture target: "
                f"missing_label_annotations=`{production_capture_deficit.get('total_missing_label_annotations', 0)}`, "
                f"recommended_label_review_rows=`{production_capture_deficit.get('recommended_label_review_rows', 0)}`, "
                f"coverage_deficits=`{production_capture_deficit.get('coverage_deficit_count', 0)}`, "
                f"next_batches=`{len(production_batches)}`, "
                f"matrix_rows=`{production_matrix_rows}`."
            )
        if capture_deficit:
            work_order = handoff.get("capture_work_order") or {}
            work_order_path = work_order.get("path") or "qa/video_eval/results/apron_harness_capture_work_order.md"
            work_order_state = "generated" if work_order.get("generated") else "expected"
            lines.append(f"- Capture work order: `{work_order_path}` ({work_order_state}).")
            matrix_csv = handoff.get("capture_matrix_csv") or {}
            matrix_csv_path = matrix_csv.get("path") or "qa/video_eval/results/apron_harness_capture_matrix.csv"
            matrix_csv_state = "generated" if matrix_csv.get("generated") else "expected"
            if matrix_csv.get("row_count") is not None:
                lines.append(
                    f"- Capture matrix CSV: `{matrix_csv_path}` "
                    f"({matrix_csv_state}, rows=`{matrix_csv.get('row_count', 0)}`)."
                )
            matrix_manifest = handoff.get("capture_matrix_manifest") or {}
            if matrix_manifest.get("path"):
                matrix_manifest_state = "generated" if matrix_manifest.get("generated") else "expected"
                lines.append(
                    f"- Capture matrix manifest: `{matrix_manifest.get('path')}` "
                    f"({matrix_manifest_state}, rows=`{matrix_manifest.get('row_count', 0)}`)."
                )
            production_matrix_csv = handoff.get("production_capture_matrix_csv") or {}
            production_matrix_csv_path = (
                production_matrix_csv.get("path")
                or "qa/video_eval/results/apron_harness_production_capture_matrix.csv"
            )
            production_matrix_csv_state = "generated" if production_matrix_csv.get("generated") else "expected"
            if production_matrix_csv.get("row_count") is not None:
                lines.append(
                    f"- Production capture matrix CSV: `{production_matrix_csv_path}` "
                    f"({production_matrix_csv_state}, rows=`{production_matrix_csv.get('row_count', 0)}`)."
                )
            production_matrix_manifest = handoff.get("production_capture_matrix_manifest") or {}
            if production_matrix_manifest.get("path"):
                production_manifest_state = (
                    "generated" if production_matrix_manifest.get("generated") else "expected"
                )
                lines.append(
                    f"- Production capture matrix manifest: `{production_matrix_manifest.get('path')}` "
                    f"({production_manifest_state}, rows=`{production_matrix_manifest.get('row_count', 0)}`)."
                )
            production_sidecar_validation = handoff.get("production_capture_matrix_sidecar_validation") or {}
            if production_sidecar_validation.get("path"):
                if production_sidecar_validation.get("checked") is True:
                    sidecar_state = "pass" if production_sidecar_validation.get("valid") else "failed"
                else:
                    sidecar_state = "not_checked"
                lines.append(f"- Production capture matrix sidecar gate: `{sidecar_state}`.")
            label_review_sidecar_validation = handoff.get("label_review_import_sidecar_validation") or {}
            if label_review_sidecar_validation.get("path"):
                if label_review_sidecar_validation.get("checked") is True:
                    label_sidecar_state = "pass" if label_review_sidecar_validation.get("valid") else "failed"
                else:
                    label_sidecar_state = "not_checked"
                lines.append(
                    "- Label-review import sidecar gate: "
                    f"`{label_sidecar_state}`; command exit `0` is not enough, "
                    "the emitted `.label_review_import.json` sidecar must validate after import."
                )
            progress = handoff.get("capture_matrix_progress") or {}
            if progress.get("row_count") is not None:
                reconciliation = progress.get("manifest_reconciliation") or {}
                manifest_counts = "not_checked"
                if reconciliation.get("checked"):
                    manifest_counts = "pass" if reconciliation.get("gate_passed") else "failed"
                lines.append(
                    "- Capture progress: "
                    f"gate=`{'pass' if progress.get('gate_passed') else 'blocked'}`, "
                    f"ready_rows=`{progress.get('ready_rows', 0)}/{progress.get('row_count', 0)}`, "
                    f"labeled_examples=`{progress.get('labeled_examples', 0)}`, "
                    f"missing_labeled_examples=`{progress.get('missing_labeled_examples', 0)}`, "
                    f"unapproved_rows=`{progress.get('unapproved_rows', 0)}`, "
                    f"manifest_counts=`{manifest_counts}`."
                )
                capability_progress = progress.get("capabilities")
                if isinstance(capability_progress, dict) and capability_progress:
                    parts = []
                    for capability in sorted(capability_progress):
                        item = capability_progress.get(capability)
                        if not isinstance(item, dict):
                            continue
                        parts.append(
                            f"{capability}(ready={item.get('ready_rows', 0)}/"
                            f"{item.get('row_count', 0)}, "
                            f"labeled={item.get('labeled_examples', 0)}, "
                            f"missing={item.get('missing_labeled_examples', 0)})"
                        )
                    if parts:
                        lines.append(f"- Capture progress by capability: {', '.join(parts)}.")
                blocked_rows = _format_capture_progress_blocked_rows(progress)
                if blocked_rows:
                    lines.append(f"- Capture next blocked rows: {blocked_rows}.")
            production_progress = handoff.get("production_capture_matrix_progress") or {}
            if production_progress.get("row_count") is not None:
                production_reconciliation = production_progress.get("manifest_reconciliation") or {}
                production_manifest_counts = "not_checked"
                if production_reconciliation.get("checked"):
                    production_manifest_counts = (
                        "pass" if production_reconciliation.get("gate_passed") else "failed"
                    )
                lines.append(
                    "- Production capture progress: "
                    f"gate=`{'pass' if production_progress.get('gate_passed') else 'blocked'}`, "
                    f"ready_rows=`{production_progress.get('ready_rows', 0)}/"
                    f"{production_progress.get('row_count', 0)}`, "
                    f"labeled_examples=`{production_progress.get('labeled_examples', 0)}`, "
                    f"missing_labeled_examples=`{production_progress.get('missing_labeled_examples', 0)}`, "
                    f"unapproved_rows=`{production_progress.get('unapproved_rows', 0)}`, "
                    f"manifest_counts=`{production_manifest_counts}`."
                )
                production_capability_progress = production_progress.get("capabilities")
                if isinstance(production_capability_progress, dict) and production_capability_progress:
                    parts = []
                    for capability in sorted(production_capability_progress):
                        item = production_capability_progress.get(capability)
                        if not isinstance(item, dict):
                            continue
                        parts.append(
                            f"{capability}(ready={item.get('ready_rows', 0)}/"
                            f"{item.get('row_count', 0)}, "
                            f"labeled={item.get('labeled_examples', 0)}, "
                            f"missing={item.get('missing_labeled_examples', 0)})"
                        )
                    if parts:
                        lines.append(
                            f"- Production capture progress by capability: {', '.join(parts)}."
                        )
                production_blocked_rows = _format_capture_progress_blocked_rows(production_progress)
                if production_blocked_rows:
                    lines.append(
                        f"- Production capture next blocked rows: {production_blocked_rows}."
                    )
    blockers = [str(blocker) for blocker in report.get("production_blockers") or []]
    next_actions = report.get("next_actions") if isinstance(report.get("next_actions"), list) else []
    if next_actions:
        action_parts = []
        evidence_parts = []
        artifact_parts = []
        for action in next_actions[:4]:
            if not isinstance(action, dict):
                continue
            priority = action.get("priority", "unknown")
            action_id = action.get("id", "unknown")
            title = str(action.get("title") or action.get("next_action") or "").strip()
            action_parts.append(f"{priority}:{action_id}({title})")
            summaries = action.get("evidence_contract_summary")
            if isinstance(summaries, list) and summaries:
                summary = str(summaries[0]).strip()
                if summary:
                    evidence_parts.append(f"{priority}:{action_id}={summary}")
            artifacts = [
                str(artifact).strip()
                for artifact in action.get("artifacts") or []
                if str(artifact).strip()
            ]
            if artifacts:
                artifact_parts.append(
                    f"{priority}:{action_id}={', '.join(artifacts)}"
                )
        if action_parts:
            lines.append(f"- Next production actions: {'; '.join(action_parts)}.")
        if evidence_parts:
            lines.append(f"- Next production evidence: {'; '.join(evidence_parts)}.")
        if artifact_parts:
            lines.append(f"- Next production artifacts: {'; '.join(artifact_parts)}.")
    if blockers:
        lines.append(f"- Production blockers: {', '.join(blockers)}.")
    errors = [str(error) for error in report.get("errors") or []]
    warnings = [str(warning) for warning in report.get("warnings") or []]
    if errors:
        lines.append(f"- Errors: {'; '.join(errors[:5])}{'; ...' if len(errors) > 5 else ''}")
    if warnings:
        lines.append(f"- Warnings: {'; '.join(warnings[:5])}{'; ...' if len(warnings) > 5 else ''}")
    return lines


def _format_capture_progress_blocked_rows(
    progress: dict[str, Any],
    *,
    max_rows_per_capability: int = 2,
    max_blockers_per_row: int = 3,
) -> str:
    rows = progress.get("rows") if isinstance(progress.get("rows"), list) else []
    if not rows:
        return ""
    by_capability: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        blockers = [
            str(blocker)
            for blocker in row.get("blockers") or []
            if str(blocker)
        ]
        missing = row.get("missing_labeled_examples")
        is_blocked = bool(blockers)
        try:
            is_blocked = is_blocked or int(missing or 0) > 0
        except (TypeError, ValueError):
            pass
        if not is_blocked:
            continue
        capability = str(row.get("target_capability") or row.get("capability") or "unknown")
        if len(by_capability.get(capability, [])) >= max_rows_per_capability:
            continue
        row_id = str(row.get("row_id") or row.get("id") or "unknown")
        blocker_text = ", ".join(blockers[:max_blockers_per_row]) or "blocked"
        if len(blockers) > max_blockers_per_row:
            blocker_text += ", ..."
        by_capability.setdefault(capability, []).append(
            f"{row_id}(missing={missing if missing is not None else 'unknown'}, blockers={blocker_text})"
        )
    parts = [
        f"{capability}={'; '.join(by_capability[capability])}"
        for capability in sorted(by_capability)
        if by_capability[capability]
    ]
    return " | ".join(parts)


def _training_preflight_status(handoff: dict[str, Any]) -> str:
    preflight = handoff.get("training_capture_preflight") or {}
    if preflight.get("checked"):
        return "pass" if preflight.get("gate_passed") else "failed"
    if preflight.get("required"):
        return "missing"
    return "not_required_for_dry_run"


def _closed_set_training_ready_status(handoff: dict[str, Any]) -> str:
    readiness = handoff.get("training_readiness")
    if isinstance(readiness, dict) and readiness.get("status"):
        return str(readiness["status"])
    if _training_preflight_status(handoff) == "pass" and handoff.get("training_dry_run_status") == "ready_to_train":
        return "ready_to_train"
    if _training_preflight_status(handoff) in {"failed", "missing"}:
        return "blocked"
    if handoff.get("training_dry_run_status"):
        return "dry_run_only"
    return "unknown"


def format_code_list(values: list[Any]) -> str:
    if not values:
        return ""
    return ", ".join(f"`{value}`" for value in values)


def format_delivery_summary(summary: dict[str, dict[str, Any]]) -> str:
    if not summary:
        return "none"
    parts = []
    for output_id in sorted(summary):
        statuses = summary[output_id].get("statuses") or {}
        status_text = "/".join(f"{status}:{count}" for status, count in sorted(statuses.items()))
        parts.append(f"{output_id}({status_text})")
    return ", ".join(parts)


def format_delivery_evidence(evidence: dict[str, Any]) -> str:
    parts = [format_delivery_summary(evidence.get("delivery_summary") or {})]
    capture = evidence.get("webhook_capture")
    if isinstance(capture, dict):
        parts.append(f"webhook_capture(requests:{capture.get('request_count', 0)})")
    smtp_capture = evidence.get("smtp_capture")
    if isinstance(smtp_capture, dict):
        parts.append(f"smtp_capture(messages:{smtp_capture.get('message_count', 0)})")
    return ", ".join(part for part in parts if part and part != "none") or "none"


def format_analytics_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "none"
    parts = []
    class_counts = summary.get("class_counts")
    if isinstance(class_counts, dict) and class_counts:
        counts = "/".join(
            f"{class_name}:{count}"
            for class_name, count in sorted(class_counts.items())
        )
        parts.append(f"class_counts({counts})")
    occupancy = summary.get("occupancy")
    if isinstance(occupancy, dict):
        parts.append(
            "occupancy("
            f"occupied:{occupancy.get('occupied_count', 0)}/{occupancy.get('zone_count', 0)}, "
            f"sample:{occupancy.get('sample_seconds', 0)}s, "
            f"max_occupied:{occupancy.get('max_longest_occupied_seconds', 0)}s, "
            f"duration_ready:{str(occupancy.get('duration_ready', False)).lower()}"
            ")"
        )
    queue = summary.get("queue")
    if isinstance(queue, dict):
        density_fragment = ""
        if queue.get("max_density_people_per_square_meter") is not None:
            density_fragment = f", density:{queue.get('max_density_people_per_square_meter')}p/m2"
        wait_fragment = ""
        if queue.get("wait_tracking_enabled"):
            wait_fragment = (
                f", wait_max:{queue.get('max_wait_seconds', 0)}s"
                f", wait_ready:{str(queue.get('wait_time_ready', False)).lower()}"
            )
        parts.append(
            "queue("
            f"persons:{queue.get('person_count', 0)}, "
            f"max_zone:{queue.get('max_zone_count', 0)}, "
            f"active:{str(queue.get('queue_active', False)).lower()}, "
            f"active_seconds:{queue.get('active_seconds', 0)}, "
            f"duration_ready:{str(queue.get('duration_ready', False)).lower()}, "
            f"calibrated:{str(queue.get('calibrated', False)).lower()}"
            f"{density_fragment}"
            f"{wait_fragment}"
            ")"
        )
    obstruction = summary.get("obstruction")
    if isinstance(obstruction, dict):
        classes = "/".join(obstruction.get("observed_classes") or [])
        density_fragment = ""
        if obstruction.get("max_density_objects_per_square_meter") is not None:
            density_fragment = f", density:{obstruction.get('max_density_objects_per_square_meter')}objects/m2"
        detail = (
            f"objects:{obstruction.get('object_count', 0)}, "
            f"max_zone:{obstruction.get('max_zone_count', 0)}, "
            f"active:{str(obstruction.get('obstruction_active', False)).lower()}, "
            f"active_seconds:{obstruction.get('active_seconds', 0)}, "
            f"duration_ready:{str(obstruction.get('duration_ready', False)).lower()}, "
            f"calibrated:{str(obstruction.get('calibrated', False)).lower()}, "
            f"severity:{obstruction.get('max_severity', 'none')}"
            f"{density_fragment}"
        )
        if classes:
            detail += f", classes:{classes}"
        parts.append(f"obstruction({detail})")
    object_lifecycle = summary.get("object_lifecycle")
    if isinstance(object_lifecycle, dict):
        classes = "/".join(object_lifecycle.get("observed_classes") or [])
        detail = (
            f"removals:{object_lifecycle.get('removal_count', 0)}, "
            f"dwell:{object_lifecycle.get('dwell_count', 0)}, "
            f"max_present:{object_lifecycle.get('max_present_seconds', 0)}s, "
            f"seen_zones:{object_lifecycle.get('seen_zone_count', 0)}, "
            f"removed:{str(object_lifecycle.get('removal_detected', False)).lower()}, "
            f"dwell_ready:{str(object_lifecycle.get('dwell_ready', False)).lower()}"
        )
        if classes:
            detail += f", classes:{classes}"
        parts.append(f"object_lifecycle({detail})")
    schedule = summary.get("schedule")
    if isinstance(schedule, dict):
        suppressed = "/".join(schedule.get("suppressed_capabilities") or [])
        invocations = "/".join(
            f"{model_key}:{count}"
            for model_key, count in sorted((schedule.get("model_invocations") or {}).items())
        )
        detail = f"suppressed:{suppressed or 'none'}, invocations:{invocations or 'none'}"
        parts.append(f"schedule({detail})")
    plate_reads = summary.get("plate_reads")
    if isinstance(plate_reads, dict):
        plates = "/".join(plate_reads.get("normalized_plates") or [])
        event_types = "/".join(plate_reads.get("event_types") or [])
        detail = f"count:{plate_reads.get('read_count', 0)}"
        if plates:
            detail += f", plates:{plates}"
        if event_types:
            detail += f", events:{event_types}"
        parts.append(f"plate_reads({detail})")
    return ", ".join(parts) if parts else "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rakshak Lens video sales-readiness evaluations.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one scenario from the manifest.")
    run.add_argument("--scenario", required=True)
    run.add_argument(
        "--skip-apply",
        action="store_true",
        help="Validate and plan only; assume the site YAML was already applied before backend startup.",
    )

    sub.add_parser("report", help="Generate markdown reports from result JSON files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    if args.command == "run":
        result = run_scenario(manifest_path, args.scenario, skip_apply=args.skip_apply)
        print(json.dumps({"scenario_id": result["scenario_id"], "status": result["status"]}, indent=2))
        return 0 if result["status"] == "ready_to_sell" else 1

    sales_path, claims_path = report(manifest_path)
    print(f"Wrote {sales_path}")
    print(f"Wrote {claims_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
