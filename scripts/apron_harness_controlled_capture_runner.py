#!/usr/bin/env python3
"""Plan or run guarded apron/harness controlled-capture label review steps."""

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
from apron_harness_train import validate_label_review_import_sidecar  # noqa: E402


DEFAULT_READINESS_REPORT = DEFAULT_RESULT_DIR / "apron_harness_readiness_doctor.json"
PLACEHOLDER_SEED_IMPORT = "/path/to/filled/apron_harness_seed_import_manifest.yaml"
PLACEHOLDER_STARTER_LABEL_REVIEW = "/path/to/filled/apron_harness_production_starter_label_review.csv"
PLACEHOLDER_FULL_LABEL_REVIEW = "/path/to/filled/apron_harness_production_label_review.csv"
DEFAULT_CAPTURE_MANIFEST = "qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml"
DEFAULT_STARTER_EMIT = "qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml"
DEFAULT_FULL_EMIT = "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml"


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return "/path/to/" in value or value.startswith("/path/to")


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


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_status(name: str, path: str | None, expected_sha256: str | None) -> dict[str, Any]:
    candidate = _artifact(path or "")
    observed_sha256 = _sha256_file(candidate) if path else None
    exists = bool(observed_sha256)
    expected = str(expected_sha256 or "")
    sha_matches = bool(expected and observed_sha256 == expected)
    blockers: list[str] = []
    if not exists:
        blockers.append("missing")
    elif not sha_matches:
        blockers.append("sha_mismatch")
    return {
        "name": name,
        "path": path or "",
        "exists": exists,
        "expected_sha256": expected,
        "observed_sha256": observed_sha256,
        "sha_matches": sha_matches,
        "ok": not blockers,
        "blockers": blockers,
    }


def _controlled_artifact_status(controlled: dict[str, Any]) -> list[dict[str, Any]]:
    statuses = [
        _artifact_status(
            "production_capture_matrix",
            controlled.get("production_capture_matrix_path"),
            controlled.get("production_capture_matrix_sha256"),
        ),
        _artifact_status(
            "production_capture_matrix_sidecar",
            controlled.get("production_capture_matrix_sidecar_path"),
            controlled.get("production_capture_matrix_sidecar_sha256"),
        ),
    ]
    handoff = controlled.get("operator_handoff")
    if isinstance(handoff, dict):
        for name, key in (
            ("capture_kickoff", "capture_kickoff"),
            ("capture_work_order", "capture_work_order"),
        ):
            artifact = handoff.get(key)
            if isinstance(artifact, dict):
                statuses.append(
                    _artifact_status(name, artifact.get("path"), artifact.get("sha256"))
                )
    templates = controlled.get("label_review_templates")
    if isinstance(templates, dict):
        for key in ("full_production", "starter_production"):
            template = templates.get(key)
            if isinstance(template, dict):
                statuses.append(
                    _artifact_status(
                        f"label_review_template_{key}",
                        template.get("path"),
                        template.get("sha256"),
                    )
                )
    return statuses


def _controlled_capture(packet: dict[str, Any]) -> dict[str, Any]:
    first_unblock = packet.get("first_unblock")
    if not isinstance(first_unblock, dict):
        return {}
    controlled = first_unblock.get("controlled_capture_path")
    return controlled if isinstance(controlled, dict) else {}


def _source_recheck_artifact(packet: dict[str, Any]) -> dict[str, Any]:
    first_unblock = packet.get("first_unblock")
    if not isinstance(first_unblock, dict):
        return {}
    artifact = first_unblock.get("source_recheck_artifact")
    return artifact if isinstance(artifact, dict) else {}


def _starter_steps(packet: dict[str, Any]) -> list[dict[str, Any]]:
    controlled = _controlled_capture(packet)
    return [
        step
        for step in controlled.get("starter_execution_plan") or []
        if isinstance(step, dict)
    ]


def _full_steps(packet: dict[str, Any]) -> list[dict[str, Any]]:
    controlled = _controlled_capture(packet)
    commands = [
        str(command)
        for command in controlled.get("commands") or []
        if command
    ]
    ids = [
        "validate_production_capture_matrix",
        "validate_full_label_review_csv",
        "import_full_label_review_csv",
    ]
    return [
        {
            "step": index + 1,
            "id": ids[index] if index < len(ids) else f"command_{index + 1}",
            "command": command,
            "writes": [DEFAULT_FULL_EMIT, f"{DEFAULT_FULL_EMIT}.label_review_import.json"]
            if "--import-label-review-csv" in command
            else [],
            "pass_signal": "LABEL_REVIEW_VALIDATION: gate=pass"
            if "--validate-label-review-csv" in command
            else None,
        }
        for index, command in enumerate(commands)
    ]


def _steps(packet: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    return _starter_steps(packet) if mode == "starter" else _full_steps(packet)


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
    text: str,
    *,
    mode: str,
    label_review_csv: str | None,
    seed_import_manifest: str | None,
    capture_manifest: str | None,
    emit_updated_manifest: str | None,
) -> str:
    if label_review_csv:
        placeholder = (
            PLACEHOLDER_STARTER_LABEL_REVIEW
            if mode == "starter"
            else PLACEHOLDER_FULL_LABEL_REVIEW
        )
        text = text.replace(placeholder, label_review_csv)
    if seed_import_manifest:
        text = text.replace(PLACEHOLDER_SEED_IMPORT, seed_import_manifest)
    if capture_manifest:
        text = text.replace(DEFAULT_CAPTURE_MANIFEST, capture_manifest)
    if emit_updated_manifest:
        default_emit = DEFAULT_STARTER_EMIT if mode == "starter" else DEFAULT_FULL_EMIT
        text = text.replace(default_emit, emit_updated_manifest)
        text = text.replace(
            f"{default_emit}.label_review_import.json",
            f"{emit_updated_manifest}.label_review_import.json",
        )
    return text


def build_plan(
    packet: dict[str, Any],
    *,
    mode: str,
    label_review_csv: str | None = None,
    seed_import_manifest: str | None = None,
    capture_manifest: str | None = None,
    emit_updated_manifest: str | None = None,
) -> dict[str, Any]:
    controlled = _controlled_capture(packet)
    raw_steps = _steps(packet, mode)
    matrix_gate = {
        "gate_passed": controlled.get("ready_rows") == controlled.get("row_count")
        and controlled.get("missing_labeled_examples") == 0,
        "row_count": controlled.get("row_count"),
        "ready_rows": controlled.get("ready_rows"),
        "missing_labeled_examples": controlled.get("missing_labeled_examples"),
        "unapproved_rows": controlled.get("unapproved_rows"),
        "unsafe_storage_rows": controlled.get("unsafe_storage_rows"),
    }
    return {
        "status": controlled.get("status"),
        "mode": mode,
        "safe_to_execute_without_reviewed_csv": False,
        "label_review_csv_supplied": _path_exists(label_review_csv),
        "seed_import_manifest_supplied": _path_exists(seed_import_manifest),
        "capture_manifest_supplied": _path_exists(capture_manifest),
        "emit_updated_manifest_supplied": not _is_placeholder(emit_updated_manifest),
        "starter_capture_rows": controlled.get("starter_capture_rows") or [],
        "starter_success_criteria": controlled.get("starter_success_criteria")
        if isinstance(controlled.get("starter_success_criteria"), dict)
        else {},
        "production_capture_matrix_gate": matrix_gate,
        "source_recheck_artifact_status": _artifact_status(
            "source_recheck_artifact",
            _source_recheck_artifact(packet).get("path"),
            _source_recheck_artifact(packet).get("sha256"),
        ),
        "artifact_status": _controlled_artifact_status(controlled),
        "post_capture_evidence_checklist": controlled.get("post_capture_evidence_checklist") or [],
        "step_count": len(raw_steps),
        "steps": [
            {
                "step": step.get("step"),
                "id": step.get("id"),
                "command": _replace_paths(
                    str(step.get("command") or ""),
                    mode=mode,
                    label_review_csv=label_review_csv,
                    seed_import_manifest=seed_import_manifest,
                    capture_manifest=capture_manifest,
                    emit_updated_manifest=emit_updated_manifest,
                )
                if step.get("command")
                else None,
                "writes": [
                    _replace_paths(
                        str(item),
                        mode=mode,
                        label_review_csv=label_review_csv,
                        seed_import_manifest=seed_import_manifest,
                        capture_manifest=capture_manifest,
                        emit_updated_manifest=emit_updated_manifest,
                    )
                    for item in (step.get("writes") or [])
                ],
                "pass_signal": step.get("pass_signal"),
                "expected_boundary": step.get("expected_boundary"),
                "stop_if": step.get("stop_if"),
            }
            for step in raw_steps
        ],
    }


def _block(result: dict[str, Any], reason: str) -> None:
    result["ok"] = False
    result["blocked"] = True
    result.setdefault("blockers", []).append(reason)


def _artifact_blockers(plan: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    source_recheck = plan.get("source_recheck_artifact_status")
    if (
        isinstance(source_recheck, dict)
        and (source_recheck.get("path") or source_recheck.get("expected_sha256"))
        and source_recheck.get("ok") is not True
    ):
        reason = ",".join(str(item) for item in (source_recheck.get("blockers") or [])) or "not_ok"
        blockers.append(f"source_recheck_artifact:{reason}")
    for artifact in plan.get("artifact_status") or []:
        if not isinstance(artifact, dict):
            continue
        if not artifact.get("path") and not artifact.get("expected_sha256"):
            continue
        if artifact.get("ok") is True:
            continue
        name = artifact.get("name") or "artifact"
        reason = ",".join(str(item) for item in (artifact.get("blockers") or [])) or "not_ok"
        blockers.append(f"{name}:{reason}")
    return blockers


def _full_mode_matrix_gate_blockers(plan: dict[str, Any]) -> list[str]:
    if plan.get("mode") != "full":
        return []
    gate = plan.get("production_capture_matrix_gate")
    if not isinstance(gate, dict) or gate.get("gate_passed") is True:
        return []
    details: list[str] = []
    for key in ("missing_labeled_examples", "unapproved_rows", "unsafe_storage_rows"):
        if key in gate:
            details.append(f"{key}={gate.get(key)}")
    suffix = f" ({', '.join(details)})" if details else ""
    return [f"production_capture_matrix_gate_not_passed{suffix}"]


def execute_plan(
    packet: dict[str, Any],
    *,
    mode: str,
    label_review_csv: str | None,
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
    if _is_placeholder(label_review_csv) or not _path_exists(label_review_csv):
        _block(result, "filled non-placeholder --label-review-csv is required before label-review validation")
        return result
    if _is_placeholder(seed_import_manifest) or not _path_exists(seed_import_manifest):
        _block(result, "filled non-placeholder --seed-import-manifest is required before label-review validation")
        return result

    plan = build_plan(
        packet,
        mode=mode,
        label_review_csv=label_review_csv,
        seed_import_manifest=seed_import_manifest,
        capture_manifest=capture_manifest,
        emit_updated_manifest=emit_updated_manifest,
    )
    artifact_blockers = _artifact_blockers(plan)
    if artifact_blockers:
        _block(
            result,
            "controlled-capture required artifacts are not ready: "
            + "; ".join(artifact_blockers),
        )
        return result
    matrix_gate_blockers = _full_mode_matrix_gate_blockers(plan)
    if matrix_gate_blockers:
        _block(
            result,
            "controlled-capture full-mode packet gates are not ready: "
            + "; ".join(matrix_gate_blockers),
        )
        return result
    command_steps = [
        step
        for step in plan["steps"]
        if step.get("command")
    ]
    if not command_steps:
        result["ok"] = False
        result["blockers"].append("no controlled-capture commands found in packet")
        return result

    validation_seen = False
    for step in command_steps:
        command = str(step["command"])
        if "/path/to/" in command:
            _block(result, f"unreplaced placeholder remains in {step.get('id')}")
            return result
        run = _run_shell(command)
        run["step_id"] = step.get("id")
        result["runs"].append(run)
        output = f"{run.get('stdout') or ''}\n{run.get('stderr') or ''}"
        if run["returncode"] != 0:
            result["ok"] = False
            result["blockers"].append(f"{step.get('id')} failed")
            return result
        if "--validate-label-review-csv" in command:
            validation_seen = True
            if "LABEL_REVIEW_VALIDATION: gate=pass" not in output:
                _block(result, "LABEL_REVIEW_VALIDATION: gate=pass is required before label-review import")
                return result
        if "--import-label-review-csv" in command:
            emit_path = emit_updated_manifest or (
                DEFAULT_STARTER_EMIT if mode == "starter" else DEFAULT_FULL_EMIT
            )
            try:
                result["label_review_import_sidecar"] = validate_label_review_import_sidecar(
                    capture_manifest_path=_artifact(str(emit_path)),
                    label_review_csv=_artifact(str(label_review_csv)),
                )
            except Exception as exc:
                _block(result, f"label review import sidecar invalid after import: {exc}")
                return result
    if not validation_seen:
        result["ok"] = False
        result["blockers"].append("label-review validation command was not run")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute guarded apron/harness controlled-capture label review steps."
    )
    parser.add_argument("--packet", default=str(DEFAULT_PRODUCTION_GATE_PACKET))
    parser.add_argument("--readiness-report", default=str(DEFAULT_READINESS_REPORT))
    parser.add_argument("--mode", choices=["starter", "full"], default="starter")
    parser.add_argument("--label-review-csv", default="")
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
        mode=args.mode,
        label_review_csv=args.label_review_csv or None,
        seed_import_manifest=args.seed_import_manifest or None,
        capture_manifest=args.capture_manifest or None,
        emit_updated_manifest=args.emit_updated_manifest or None,
    )
    result: dict[str, Any] = {
        "ok": validation.get("ok") is True and not args.execute,
        "mode": "execute" if args.execute else "plan",
        "capture_mode": args.mode,
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
            mode=args.mode,
            label_review_csv=args.label_review_csv or None,
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
            "CONTROLLED_CAPTURE_RUNNER: "
            f"ok={result['ok']} mode={result['mode']} "
            f"capture_mode={result['capture_mode']} "
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
