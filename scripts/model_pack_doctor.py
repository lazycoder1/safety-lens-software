#!/usr/bin/env python3
"""Validate model-pack runtime prerequisites without loading heavy models."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import model_manager  # noqa: E402


DEFAULT_PACKS = ROOT / "qa" / "video_eval" / "model_packs.yaml"
DEFAULT_OUT = ROOT / "qa" / "video_eval" / "results" / "model_pack_device_probe.json"
LEGACY_MODEL_ARTIFACT_PATHS = [
    ROOT / "yolo11n.pt",
    ROOT / "yolo11s.pt",
    ROOT / "yolo11n-pose.pt",
    ROOT / "yolo26n.pt",
    ROOT / "yolo26m.pt",
    ROOT / "yoloe-11s-seg.pt",
    ROOT / "yoloe-26n-seg.pt",
    ROOT / "yoloe-26s-seg.pt",
    ROOT / "frontend" / "yolo11n.pt",
    ROOT / "frontend" / "yolo11s.pt",
    ROOT / "frontend" / "yolo11n-pose.pt",
    ROOT / "frontend" / "yolo26n.pt",
    ROOT / "frontend" / "yolo26m.pt",
    ROOT / "frontend" / "yoloe-11s-seg.pt",
    ROOT / "frontend" / "yoloe-26n-seg.pt",
    ROOT / "frontend" / "yoloe-26s-seg.pt",
    ROOT / "models" / "coco_primary" / "yolo11n.pt",
    ROOT / "models" / "coco_primary" / "yolo11s.pt",
    ROOT / "models" / "pose_specialist" / "yolo11n-pose.pt",
    ROOT / "models" / "yoloe_open_vocab" / "yoloe-11s-seg.pt",
    ROOT.parent / "yolo11n.pt",
    ROOT.parent / "yolo11s.pt",
    ROOT.parent / "yolo11n-pose.pt",
    ROOT.parent / "yolo26n.pt",
    ROOT.parent / "yolo26m.pt",
    ROOT.parent / "yoloe-11s-seg.pt",
    ROOT.parent / "yoloe-26n-seg.pt",
    ROOT.parent / "yoloe-26s-seg.pt",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def macos_version_status() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {
            "checked": False,
            "reason": "not_darwin",
        }
    try:
        result = subprocess.run(
            ["sw_vers"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # pragma: no cover - host dependent
        return {
            "checked": True,
            "ok": False,
            "error": str(exc),
        }

    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return {
        "checked": True,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "product_name": parsed.get("ProductName"),
        "product_version": parsed.get("ProductVersion"),
        "build_version": parsed.get("BuildVersion"),
        "raw": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            "installed": False,
            "error": str(exc),
            "mps_built": False,
            "mps_available": False,
            "mps_probe_ok": False,
            "mps_runtime_error": str(exc),
            "cuda_available": False,
        }

    mps_built = False
    mps_available = False
    mps_probe_ok = False
    mps_runtime_error = None
    try:
        mps_built = bool(torch.backends.mps.is_built())
    except Exception as exc:
        mps_runtime_error = str(exc)
    try:
        mps_available = bool(torch.backends.mps.is_available())
    except Exception as exc:
        mps_runtime_error = str(exc)
    if mps_built:
        try:
            torch.ones(1, device="mps")
            mps_probe_ok = True
        except Exception as exc:
            mps_runtime_error = str(exc)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    return {
        "installed": True,
        "version": getattr(torch, "__version__", "unknown"),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_probe_ok": mps_probe_ok,
        "mps_runtime_error": mps_runtime_error,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def model_file_status(model_key: str) -> dict[str, Any]:
    definition = model_manager.MODEL_DEFINITIONS.get(model_key)
    if not definition:
        return {"known": False}
    candidates = [definition.get("local_path"), *(definition.get("legacy_paths") or [])]
    existing = [str(path) for path in candidates if isinstance(path, Path) and path.exists()]
    return {
        "known": True,
        "filename": definition.get("filename"),
        "local_path": str(definition.get("local_path")),
        "existing_paths": existing,
        "download_url": definition.get("download_url"),
        "ready_for_local_load": bool(existing),
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def artifact_layout_status() -> dict[str, Any]:
    canonical_paths: dict[str, dict[str, Any]] = {}
    for model_key, definition in model_manager.MODEL_DEFINITIONS.items():
        local_path = definition.get("local_path")
        if not isinstance(local_path, Path):
            continue
        canonical_paths[model_key] = {
            "path": _display_path(local_path),
            "exists": local_path.exists(),
        }

    checked_paths = [_display_path(path) for path in LEGACY_MODEL_ARTIFACT_PATHS]
    unexpected_paths = [_display_path(path) for path in LEGACY_MODEL_ARTIFACT_PATHS if path.exists()]
    ok = not unexpected_paths
    return {
        "ok": ok,
        "gate": "pass" if ok else "blocked_legacy_artifacts_present",
        "canonical_model_paths": canonical_paths,
        "checked_legacy_paths": checked_paths,
        "unexpected_paths": unexpected_paths,
        "unexpected_count": len(unexpected_paths),
    }


def local_device_status(local_device: str, torch_info: dict[str, Any]) -> dict[str, Any]:
    local_device = str(local_device or "").lower()
    requires_mps = local_device in {"mps", "mps_only"} or "mps_required" in local_device
    allows_mps = "mps" in local_device
    allows_cpu_fallback = "cpu_fallback" in local_device or "cpu" in local_device
    mps_ready = bool(torch_info.get("mps_available")) and torch_info.get("mps_probe_ok", True) is not False

    if requires_mps:
        satisfied = mps_ready
        reason = "mps_available" if satisfied else "mps_required_but_unavailable"
    elif allows_mps and mps_ready:
        satisfied = True
        reason = "mps_available"
    elif allows_cpu_fallback:
        satisfied = True
        reason = "cpu_fallback_only" if not mps_ready else "mps_available_with_cpu_fallback"
    else:
        satisfied = True
        reason = "no_local_mps_requirement_declared"

    performance_gate_satisfied = mps_ready if allows_mps or requires_mps else satisfied
    if not satisfied:
        evidence_scope = "not_satisfied"
    elif performance_gate_satisfied:
        evidence_scope = "local_mps_performance" if (allows_mps or requires_mps) else "functional_runtime"
    else:
        evidence_scope = "functional_wiring_only_cpu_fallback"

    return {
        "declared_local_device": local_device,
        "satisfied": satisfied,
        "reason": reason,
        "mps_available": bool(torch_info.get("mps_available")),
        "mps_probe_ok": torch_info.get("mps_probe_ok"),
        "mps_runtime_error": torch_info.get("mps_runtime_error"),
        "cpu_fallback_allowed": allows_cpu_fallback,
        "performance_gate_satisfied": performance_gate_satisfied,
        "evidence_scope": evidence_scope,
    }


def build_report(pack_path: Path = DEFAULT_PACKS) -> dict[str, Any]:
    packs_doc = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
    torch_info = torch_status()
    macos_info = macos_version_status()
    artifact_layout = artifact_layout_status()
    pack_reports: dict[str, Any] = {}

    for pack_id, pack in (packs_doc.get("packs") or {}).items():
        registry_models = pack.get("registry_models") or {}
        model_reports = {}
        local_statuses = []
        for model_key, model_info in registry_models.items():
            if model_key not in model_manager.MODEL_DEFINITIONS:
                continue
            local_status = local_device_status(str(model_info.get("local_device") or ""), torch_info)
            local_statuses.append(local_status)
            model_reports[model_key] = {
                "declared": model_info,
                "registry": model_file_status(model_key),
                "local_device_status": local_status,
            }

        pack_reports[pack_id] = {
            "status": pack.get("status"),
            "model_keys": pack.get("model_keys") or [],
            "local_device_satisfied": all(item["satisfied"] for item in local_statuses) if local_statuses else False,
            "local_functional_satisfied": all(item["satisfied"] for item in local_statuses) if local_statuses else False,
            "local_performance_gate_satisfied": (
                all(item["performance_gate_satisfied"] for item in local_statuses) if local_statuses else False
            ),
            "local_device_reasons": sorted({item["reason"] for item in local_statuses}),
            "local_evidence_scopes": sorted({item["evidence_scope"] for item in local_statuses}),
            "models": model_reports,
        }

    mps_ready = bool(torch_info.get("mps_available")) and torch_info.get("mps_probe_ok", True) is not False
    mps_gate = "mps_available" if mps_ready else "mps_unavailable_cpu_fallback_only"
    performance_gate = "pass" if mps_ready else "blocked_cpu_fallback_only"
    return {
        "generated_at": utc_now(),
        "host": platform.node(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "macos": macos_info,
        },
        "torch": torch_info,
        "mps_acceptance_gate": mps_gate,
        "local_performance_acceptance_gate": performance_gate,
        "model_artifact_layout_gate": artifact_layout["gate"],
        "artifact_layout": artifact_layout,
        "pack_path": str(pack_path),
        "packs": pack_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe local device readiness for Rakshak Lens model packs.")
    parser.add_argument("--packs", default=str(DEFAULT_PACKS), help="Path to model_packs.yaml")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(Path(args.packs))
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
