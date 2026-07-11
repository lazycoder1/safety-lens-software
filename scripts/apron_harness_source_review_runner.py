#!/usr/bin/env python3
"""Plan or run guarded apron/harness public seed-source review steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from apron_harness_readiness_doctor import (  # noqa: E402
    DEFAULT_PRODUCTION_GATE_PACKET,
    DEFAULT_RESULT_DIR,
    _load_json,
    _rel,
    validate_production_gate_packet,
)
from apron_harness_train import validate_seed_export_import_sidecar  # noqa: E402


DEFAULT_READINESS_REPORT = DEFAULT_RESULT_DIR / "apron_harness_readiness_doctor.json"
DEFAULT_SEED_SOURCE_REVIEW_REPORT = DEFAULT_RESULT_DIR / "apron_harness_seed_source_review.json"
PLACEHOLDER_SEED_IMPORT = "/path/to/filled/apron_harness_seed_import_manifest.yaml"
PLACEHOLDER_CAPTURE_MANIFEST = "/path/to/cleared/apron_harness_capture_manifest.yaml"
PLACEHOLDER_EMIT_MANIFEST = (
    "/path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml"
)


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return "/path/to/" in value or value.startswith("/path/to")


def _step_by_id(packet: dict[str, Any], step_id: str) -> dict[str, Any]:
    first_unblock = packet.get("first_unblock")
    if not isinstance(first_unblock, dict):
        return {}
    for step in first_unblock.get("source_review_execution_plan") or []:
        if isinstance(step, dict) and step.get("id") == step_id:
            return step
    return {}


def _source_steps(packet: dict[str, Any]) -> list[dict[str, Any]]:
    first_unblock = packet.get("first_unblock")
    if not isinstance(first_unblock, dict):
        return []
    return [
        step
        for step in first_unblock.get("source_review_execution_plan") or []
        if isinstance(step, dict)
    ]


def _run_shell(command: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _replace_paths(
    command: str,
    *,
    seed_import_manifest: str | None,
    capture_manifest: str | None,
    emit_updated_manifest: str | None,
) -> str:
    if seed_import_manifest:
        command = command.replace(PLACEHOLDER_SEED_IMPORT, seed_import_manifest)
    if capture_manifest:
        command = command.replace(PLACEHOLDER_CAPTURE_MANIFEST, capture_manifest)
    if emit_updated_manifest:
        command = command.replace(PLACEHOLDER_EMIT_MANIFEST, emit_updated_manifest)
        command = command.replace(
            f"{PLACEHOLDER_EMIT_MANIFEST}.seed_export_import.json",
            f"{emit_updated_manifest}.seed_export_import.json",
        )
    return command


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifact_status(source: dict[str, Any]) -> dict[str, Any]:
    artifact_pairs = [
        ("review_packet", "review_packet_path", "review_packet_sha256"),
        ("review_evidence_template", "review_evidence_template_path", "review_evidence_template_sha256"),
        ("review_prefill", "review_prefill_path", "review_prefill_sha256"),
        ("review_checklist_csv", "review_checklist_csv_path", "review_checklist_csv_sha256"),
        (
            "seed_import_manifest_template",
            "seed_import_manifest_template_path",
            "seed_import_manifest_template_sha256",
        ),
    ]
    artifacts: dict[str, Any] = {}
    blockers: list[str] = []
    for name, path_key, sha_key in artifact_pairs:
        path = str(source.get(path_key) or "")
        expected_sha = str(source.get(sha_key) or "")
        candidate = _artifact(path) if path else ROOT
        observed_sha = _sha256_file(candidate) if path else None
        exists = bool(observed_sha)
        sha_matches = bool(expected_sha and observed_sha == expected_sha)
        artifacts[name] = {
            "path": path,
            "exists": exists,
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "sha_matches": sha_matches,
        }
        if not exists:
            blockers.append(f"{name}_missing")
        elif not sha_matches:
            blockers.append(f"{name}_sha_mismatch")
    return {
        "source_ref": source.get("source_ref"),
        "capability": source.get("capability"),
        "ok": not blockers,
        "blockers": blockers,
        "artifacts": artifacts,
    }


def _single_artifact_status(artifact: dict[str, Any] | None, *, kind: str) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {
            "kind": kind,
            "ok": False,
            "blockers": [f"{kind}_missing"],
            "path": "",
            "exists": False,
            "expected_sha256": "",
            "observed_sha256": None,
            "sha_matches": False,
        }
    path = str(artifact.get("path") or "")
    expected_sha = str(artifact.get("sha256") or "")
    observed_sha = _sha256_file(_artifact(path)) if path else None
    exists = bool(observed_sha)
    sha_matches = bool(expected_sha and observed_sha == expected_sha)
    blockers: list[str] = []
    if not exists:
        blockers.append(f"{kind}_missing")
    elif not sha_matches:
        blockers.append(f"{kind}_sha_mismatch")
    return {
        "kind": kind,
        "ok": not blockers,
        "blockers": blockers,
        "path": path,
        "exists": exists,
        "expected_sha256": expected_sha,
        "observed_sha256": observed_sha,
        "sha_matches": sha_matches,
        "evidence_boundary": artifact.get("evidence_boundary") or "",
    }


def build_plan(
    packet: dict[str, Any],
    *,
    seed_import_manifest: str | None = None,
    capture_manifest: str | None = None,
    emit_updated_manifest: str | None = None,
) -> dict[str, Any]:
    first_unblock = packet.get("first_unblock")
    if not isinstance(first_unblock, dict):
        first_unblock = {}
    steps = _source_steps(packet)
    fill_step = _step_by_id(packet, "fill_minimum_review_evidence")
    minimum_review_sources = first_unblock.get("minimum_review_sources")
    if not isinstance(minimum_review_sources, list):
        minimum_review_sources = []
    return {
        "status": first_unblock.get("status"),
        "safe_to_execute_without_approval": False,
        "seed_import_manifest_supplied": _path_exists(seed_import_manifest),
        "capture_manifest_supplied": _path_exists(capture_manifest),
        "emit_updated_manifest_supplied": not _is_placeholder(emit_updated_manifest),
        "minimum_approval_path": first_unblock.get("minimum_approval_path")
        if isinstance(first_unblock.get("minimum_approval_path"), dict)
        else {},
        "source_recheck_artifact_status": _single_artifact_status(
            first_unblock.get("source_recheck_artifact")
            if isinstance(first_unblock.get("source_recheck_artifact"), dict)
            else None,
            kind="source_recheck",
        ),
        "minimum_review_sources": minimum_review_sources,
        "minimum_review_artifact_status": [
            _source_artifact_status(source)
            for source in minimum_review_sources
            if isinstance(source, dict)
        ],
        "next_source_reviews": first_unblock.get("next_source_reviews")
        if isinstance(first_unblock.get("next_source_reviews"), list)
        else [],
        "evidence_boundary": first_unblock.get("evidence_boundary") or "",
        "required_sources": fill_step.get("required_sources") or [],
        "human_approval_stop_conditions": fill_step.get("stop_if") or [],
        "step_count": len(steps),
        "steps": [
            {
                "step": step.get("step"),
                "id": step.get("id"),
                "command": _replace_paths(
                    str(step.get("command") or ""),
                    seed_import_manifest=seed_import_manifest,
                    capture_manifest=capture_manifest,
                    emit_updated_manifest=emit_updated_manifest,
                )
                if step.get("command")
                else None,
                "writes": [
                    _replace_paths(
                        str(item),
                        seed_import_manifest=seed_import_manifest,
                        capture_manifest=capture_manifest,
                        emit_updated_manifest=emit_updated_manifest,
                    )
                    for item in (step.get("writes") or [])
                ],
                "pass_signal": step.get("pass_signal"),
                "expected_boundary": step.get("expected_boundary"),
                "required_sources": step.get("required_sources"),
            }
            for step in steps
        ],
    }


def _block(result: dict[str, Any], reason: str) -> int:
    result["ok"] = False
    result["blocked"] = True
    result.setdefault("blockers", []).append(reason)
    return 2


def _source_recheck_blockers(packet: dict[str, Any]) -> list[str]:
    first_unblock = packet.get("first_unblock")
    if not isinstance(first_unblock, dict):
        first_unblock = {}
    artifact = first_unblock.get("source_recheck_artifact")
    status = _single_artifact_status(
        artifact if isinstance(artifact, dict) else None,
        kind="source_recheck",
    )
    if not status.get("path") and not status.get("expected_sha256"):
        return []
    if status.get("ok") is True:
        return []
    return [",".join(str(item) for item in (status.get("blockers") or [])) or "not_ok"]


def _path_exists(path: str | None) -> bool:
    if _is_placeholder(path):
        return False
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.exists()


def _artifact(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def execute_plan(
    packet: dict[str, Any],
    *,
    seed_import_manifest: str | None,
    capture_manifest: str | None,
    emit_updated_manifest: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "blocked": False,
        "blockers": [],
        "runs": [],
    }
    source_recheck_blockers = _source_recheck_blockers(packet)
    if source_recheck_blockers:
        _block(
            result,
            "source-review source-recheck artifact is not ready: "
            + "; ".join(source_recheck_blockers),
        )
        return result
    validate_bundle = _step_by_id(packet, "validate_source_review_bundle")
    validate_bundle_command = validate_bundle.get("command")
    if not validate_bundle_command:
        result["ok"] = False
        result["blockers"].append("validate_source_review_bundle command missing")
        return result
    bundle_run = _run_shell(str(validate_bundle_command))
    bundle_run["step_id"] = "validate_source_review_bundle"
    result["runs"].append(bundle_run)
    bundle_output = f"{bundle_run.get('stdout') or ''}\n{bundle_run.get('stderr') or ''}"
    if bundle_run["returncode"] != 0 or "REVIEW_BUNDLE: ok=True" not in bundle_output:
        result["ok"] = False
        result["blockers"].append("REVIEW_BUNDLE: ok=True is required before source review execution")
        return result

    if _is_placeholder(seed_import_manifest) or not _path_exists(seed_import_manifest):
        _block(
            result,
            "filled non-placeholder --seed-import-manifest is required before validating or importing public seed exports",
        )
        return result

    validate_import = _step_by_id(packet, "validate_seed_import_manifest")
    validate_import_command = _replace_paths(
        str(validate_import.get("command") or ""),
        seed_import_manifest=seed_import_manifest,
        capture_manifest=capture_manifest,
        emit_updated_manifest=emit_updated_manifest,
    )
    if not validate_import_command:
        result["ok"] = False
        result["blockers"].append("validate_seed_import_manifest command missing")
        return result
    import_run = _run_shell(validate_import_command)
    import_run["step_id"] = "validate_seed_import_manifest"
    result["runs"].append(import_run)
    import_output = f"{import_run.get('stdout') or ''}\n{import_run.get('stderr') or ''}"
    if import_run["returncode"] != 0 or "IMPORT_MANIFEST: gate=pass" not in import_output:
        _block(result, "IMPORT_MANIFEST: gate=pass is required before seed export materialization")
        return result

    if _is_placeholder(capture_manifest) or not _path_exists(capture_manifest):
        _block(result, "cleared existing --capture-manifest is required before seed export materialization")
        return result
    if _is_placeholder(emit_updated_manifest):
        _block(result, "non-placeholder --emit-updated-manifest is required before seed export materialization")
        return result

    materialize = _step_by_id(packet, "materialize_approved_seed_exports")
    materialize_command = _replace_paths(
        str(materialize.get("command") or ""),
        seed_import_manifest=seed_import_manifest,
        capture_manifest=capture_manifest,
        emit_updated_manifest=emit_updated_manifest,
    )
    if not materialize_command:
        result["ok"] = False
        result["blockers"].append("materialize_approved_seed_exports command missing")
        return result
    materialize_run = _run_shell(materialize_command)
    materialize_run["step_id"] = "materialize_approved_seed_exports"
    result["runs"].append(materialize_run)
    if materialize_run["returncode"] != 0:
        result["ok"] = False
        result["blockers"].append("approved seed export materialization failed")
        return result
    try:
        result["seed_export_import_sidecar"] = validate_seed_export_import_sidecar(
            capture_manifest_path=_artifact(str(emit_updated_manifest)),
            seed_import_manifest=_artifact(str(seed_import_manifest)),
            seed_source_review_report=DEFAULT_SEED_SOURCE_REVIEW_REPORT,
        )
    except Exception as exc:
        _block(result, f"seed export import sidecar invalid after materialization: {exc}")
        return result

    rerun = _step_by_id(packet, "rerun_readiness_packet")
    rerun_command = rerun.get("command")
    if rerun_command:
        rerun_result = _run_shell(str(rerun_command))
        rerun_result["step_id"] = "rerun_readiness_packet"
        result["runs"].append(rerun_result)
        if rerun_result["returncode"] != 0:
            result["ok"] = False
            result["blockers"].append("readiness packet rerun failed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute guarded apron/harness source-review import steps."
    )
    parser.add_argument("--packet", default=str(DEFAULT_PRODUCTION_GATE_PACKET))
    parser.add_argument("--readiness-report", default=str(DEFAULT_READINESS_REPORT))
    parser.add_argument("--seed-import-manifest", default="")
    parser.add_argument("--capture-manifest", default="")
    parser.add_argument("--emit-updated-manifest", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", help="Optional path to write the runner result JSON.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    packet_path = Path(args.packet)
    readiness_path = Path(args.readiness_report)
    validation = validate_production_gate_packet(packet_path, readiness_report_path=readiness_path)
    packet = _load_json(packet_path)
    plan = build_plan(
        packet,
        seed_import_manifest=args.seed_import_manifest or None,
        capture_manifest=args.capture_manifest or None,
        emit_updated_manifest=args.emit_updated_manifest or None,
    )
    result: dict[str, Any] = {
        "ok": validation.get("ok") is True and not args.execute,
        "mode": "execute" if args.execute else "plan",
        "packet": _rel(packet_path),
        "readiness_report": _rel(readiness_path),
        "packet_validation_ok": validation.get("ok") is True,
        "packet_validation_errors": validation.get("errors") or [],
        "plan": plan,
        "blocked": False,
        "blockers": [],
        "runs": [],
    }
    if validation.get("ok") is not True:
        result["ok"] = False
    elif args.execute:
        execution = execute_plan(
            packet,
            seed_import_manifest=args.seed_import_manifest or None,
            capture_manifest=args.capture_manifest or None,
            emit_updated_manifest=args.emit_updated_manifest or None,
        )
        result.update(execution)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result["out"] = _rel(out_path)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            "SOURCE_REVIEW_RUNNER: "
            f"ok={result['ok']} mode={result['mode']} "
            f"packet_ok={result['packet_validation_ok']} "
            f"steps={plan['step_count']} blocked={result['blocked']}"
        )
        for error in result["packet_validation_errors"]:
            print(f"ERROR: {error}")
        for blocker in result["blockers"]:
            print(f"BLOCKED: {blocker}")
        for step in plan["steps"]:
            print(f"- {step['id']}")
            if step.get("command"):
                print(f"  command: {step['command']}")
    if result["ok"]:
        return 0
    if result.get("blocked"):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
