#!/usr/bin/env python3
"""Rakshak Lens local runtime doctor.

Checks the dev stack from the outside, using the same HTTP surfaces the UI uses.
It is intentionally read-only except for creating a short-lived JWT locally.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

FRONTEND_URL = os.environ.get("SAFETYLENS_FRONTEND_URL", "http://127.0.0.1:3030").rstrip("/")
BACKEND_URL = os.environ.get("SAFETYLENS_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_SERVER_URL = os.environ.get("SAFETYLENS_MODEL_SERVER_URL", "http://127.0.0.1:8100").rstrip("/")
TIMEOUT_SECONDS = float(os.environ.get("SAFETYLENS_DOCTOR_TIMEOUT_SECONDS", "5"))


@dataclass
class Check:
    name: str
    status: str
    detail: str
    elapsed_ms: float | None = None
    data: Any = None


def _admin_token() -> str:
    import auth_store

    return auth_store.create_token("runtime-doctor", "runtime-doctor", "admin")


def _request_json(url: str, token: str | None = None, timeout: float = TIMEOUT_SECONDS) -> tuple[int, Any, float]:
    headers = {"Accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                return resp.status, json.loads(raw.decode("utf-8") or "{}"), elapsed_ms
            return resp.status, raw.decode("utf-8", errors="replace")[:500], elapsed_ms
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        raw = exc.read()
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = raw.decode("utf-8", errors="replace")[:500]
        return exc.code, body, elapsed_ms


def _check_endpoint(name: str, url: str, token: str | None = None, ok_statuses: set[int] | None = None) -> Check:
    ok_statuses = ok_statuses or {200}
    try:
        status, body, elapsed_ms = _request_json(url, token=token)
        if status in ok_statuses:
            return Check(name, "PASS", f"HTTP {status}", elapsed_ms, body)
        return Check(name, "FAIL", f"HTTP {status}: {_summarize_body(body)}", elapsed_ms, body)
    except Exception as exc:
        return Check(name, "FAIL", str(exc))


def _summarize_body(body: Any) -> str:
    if isinstance(body, dict):
        if body.get("detail"):
            return str(body["detail"])
        if body.get("message"):
            return str(body["message"])
        return json.dumps(body)[:180]
    return str(body)[:180]


def _as_list(value: Any, key: str) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get(key), list):
        return [item for item in value[key] if isinstance(item, dict)]
    return []


def _count_cameras(health: dict | None, cameras_response: Any) -> dict[str, int]:
    health_cameras = _as_list(health or {}, "cameras")
    api_cameras = _as_list(cameras_response, "cameras")
    cameras = health_cameras or api_cameras
    return {
        "total": len(cameras),
        "enabled": sum(1 for cam in cameras if cam.get("enabled", True)),
        "running": sum(1 for cam in cameras if cam.get("workerRunning") or cam.get("runtimeStatus") == "running"),
        "awaiting": sum(1 for cam in cameras if cam.get("runtimeStatus") == "awaiting_model_install"),
        "with_frame": sum(1 for cam in cameras if cam.get("frameAvailable")),
    }


def _models_ready(model_payload: Any) -> tuple[int, int]:
    models = _as_list(model_payload, "models")
    return sum(1 for model in models if model.get("is_ready")), len(models)


def _recent_errors(errors_payload: Any) -> list[dict]:
    errors = _as_list(errors_payload, "errors")
    if not errors and isinstance(errors_payload, list):
        errors = _as_list(errors_payload, "errors")
    return errors[:5]


def _print_check(check: Check):
    elapsed = f" ({check.elapsed_ms:.0f} ms)" if check.elapsed_ms is not None else ""
    print(f"[{check.status}] {check.name}: {check.detail}{elapsed}")


def main() -> int:
    token = _admin_token()
    checks: list[Check] = [
        _check_endpoint("Frontend", f"{FRONTEND_URL}/"),
        _check_endpoint("Backend health", f"{BACKEND_URL}/api/health"),
        _check_endpoint("Model server health", f"{MODEL_SERVER_URL}/api/health"),
        _check_endpoint("Cameras API", f"{BACKEND_URL}/api/cameras", token=token),
        _check_endpoint("Alerts API", f"{BACKEND_URL}/api/alerts?limit=50", token=token),
        _check_endpoint("Models API", f"{BACKEND_URL}/api/models", token=token),
        _check_endpoint("Alert outputs API", f"{BACKEND_URL}/api/alert-outputs", token=token),
        _check_endpoint("Recent errors API", f"{BACKEND_URL}/api/errors?limit=10", token=token),
    ]

    for check in checks:
        _print_check(check)

    health = checks[1].data if checks[1].status == "PASS" and isinstance(checks[1].data, dict) else {}
    model_health = checks[2].data if checks[2].status == "PASS" and isinstance(checks[2].data, dict) else {}
    cameras = _count_cameras(health, checks[3].data)
    ready, total = _models_ready(checks[5].data)
    errors = _recent_errors(checks[7].data)

    print("")
    print("Summary")
    print(f"- Backend status: {health.get('status', 'unknown')}")
    if health.get("reasons"):
        print(f"- Backend reasons: {', '.join(health['reasons'])}")
    print(f"- Model server: {model_health.get('status', 'unknown')} ({model_health.get('models_ready', 0)}/{model_health.get('models_total', 0)} ready)")
    print(f"- Backend model API: {ready}/{total} ready")
    print(
        "- Cameras: "
        f"{cameras['running']}/{cameras['enabled']} running, "
        f"{cameras['with_frame']} with frames, "
        f"{cameras['awaiting']} awaiting models"
    )

    warnings: list[str] = []
    failures = [check for check in checks if check.status == "FAIL"]
    if cameras["awaiting"] and model_health.get("models_ready", 0):
        warnings.append("Cameras are awaiting models even though the model server is ready; backend retry should recover this within 10 seconds.")
    if cameras["enabled"] and cameras["running"] < cameras["enabled"]:
        warnings.append("One or more enabled cameras are not running.")
    slow = [check for check in checks if check.elapsed_ms is not None and check.elapsed_ms > 2000]
    if slow:
        warnings.append("Slow endpoints: " + ", ".join(f"{check.name} {check.elapsed_ms:.0f} ms" for check in slow))
    for err in errors:
        message = str(err.get("message") or "")
        url = str(err.get("url") or "")
        if "ws://localhost:3030/ws" in message or "ws://localhost:3030/ws" in url:
            warnings.append("Recent frontend WebSocket error points at Vite/browser dev connection instability.")
            break

    if errors:
        print("- Recent errors:")
        for err in errors:
            source = err.get("source", "unknown")
            message = str(err.get("message") or "no message").replace("\n", " ")[:140]
            print(f"  {source}: {message}")

    if warnings:
        print("")
        print("Warnings")
        for warning in warnings:
            print(f"- {warning}")

    if failures:
        print("")
        print("Next action")
        print("- Restart with ./dev.sh, then rerun this doctor command.")
        return 1
    if warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
