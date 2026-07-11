#!/usr/bin/env python3
"""Run the closed-set apron/harness candidate scenarios from the gate packet."""

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
    PLANNED_MODEL_KEY,
    _artifact_path,
    _candidate_runtime_evidence_status,
    _load_json,
    _rel,
    validate_production_gate_packet,
)


DEFAULT_READINESS_REPORT = DEFAULT_RESULT_DIR / "apron_harness_readiness_doctor.json"
DEFAULT_PREFLIGHT_REFRESH_REPORT = (
    DEFAULT_RESULT_DIR / "apron_harness_candidate_runtime_preflight_refresh.json"
)


def _model_ready(packet: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    registry = packet.get("model_registry_handoff")
    if not isinstance(registry, dict):
        registry = {}
    destination_path = registry.get("destination_path") or (
        "models/ppe_closed_set_candidate/apron-harness-ppe.onnx"
    )
    metadata_path = f"{destination_path}.registry.json"
    destination = _artifact_path(str(destination_path))
    metadata = _artifact_path(metadata_path)
    if registry.get("registry_status") != "registered":
        errors.append(f"registry_status is {registry.get('registry_status')!r}, not 'registered'")
    if registry.get("destination_exists") is not True or not destination.exists():
        errors.append(f"{destination_path} is missing")
    if registry.get("metadata_valid") is not True or not metadata.exists():
        errors.append(f"{metadata_path} is missing or invalid")
    return not errors, errors


def _step_commands(step: dict[str, Any]) -> list[tuple[str, str]]:
    commands = step.get("commands")
    if not isinstance(commands, dict):
        return []
    ordered = []
    for name in ("backup", "validate", "plan", "apply", "run", "restore"):
        command = commands.get(name)
        if command:
            ordered.append((name, str(command)))
    return ordered


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


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_status(name: str, path: str | None, *, output: bool = False) -> dict[str, Any]:
    candidate = _artifact_path(str(path or ""))
    observed_sha256 = _sha256_file(candidate) if path else None
    exists = bool(observed_sha256)
    return {
        "name": name,
        "path": str(path or ""),
        "exists": exists,
        "sha256": observed_sha256,
        "output": output,
        "ok": exists or output,
        "blockers": [] if exists or output else ["missing"],
    }


def build_plan(packet: dict[str, Any]) -> dict[str, Any]:
    runtime_plan = packet.get("candidate_runtime_execution_plan")
    if not isinstance(runtime_plan, dict):
        runtime_plan = {}
    steps = [
        step
        for step in runtime_plan.get("steps") or []
        if isinstance(step, dict)
    ]
    ready, model_errors = _model_ready(packet)
    success_criteria = runtime_plan.get("success_criteria")
    return {
        "model_ready": ready,
        "model_errors": model_errors,
        "required_model_key": runtime_plan.get("required_model_key") or PLANNED_MODEL_KEY,
        "one_detection_at_a_time": runtime_plan.get("one_detection_at_a_time") is True,
        "scenario_order": runtime_plan.get("scenario_order")
        if isinstance(runtime_plan.get("scenario_order"), list)
        else [],
        "success_criteria": success_criteria
        if isinstance(success_criteria, (dict, list))
        else [],
        "runbook": runtime_plan.get("runbook") if isinstance(runtime_plan.get("runbook"), dict) else {},
        "step_count": len(steps),
        "steps": [
            {
                "step": step.get("step"),
                "scenario_id": step.get("scenario_id"),
                "capability": step.get("capability"),
                "role": step.get("role"),
                "expected_result_path": str(step.get("expected_result_path") or ""),
                "artifact_status": [
                    _artifact_status("config", step.get("config_path")),
                    _artifact_status(
                        "expected_result",
                        step.get("expected_result_path"),
                        output=True,
                    ),
                ],
                "current_result": step.get("current_result")
                if isinstance(step.get("current_result"), dict)
                else {},
                "commands": [
                    {"name": name, "command": command}
                    for name, command in _step_commands(step)
                ],
            }
            for step in steps
        ],
    }


def _validate_step_result(step: dict[str, Any]) -> dict[str, Any]:
    template = {
        "scenario_id": step.get("scenario_id"),
        "capability": step.get("capability"),
        "role": step.get("role"),
        "expected_result_path": step.get("expected_result_path"),
    }
    status = _candidate_runtime_evidence_status([template])
    rows = status.get("results") if isinstance(status.get("results"), list) else []
    row = rows[0] if rows and isinstance(rows[0], dict) else {}
    return {
        "ok": status.get("valid") is True,
        "blockers": status.get("blockers") or [],
        "result": row,
    }


def _validate_step_missing_model_preflight(step: dict[str, Any]) -> dict[str, Any]:
    evidence_validation = _validate_step_result(step)
    row = evidence_validation.get("result")
    if not isinstance(row, dict):
        row = {}
    errors = {str(error) for error in (row.get("errors") or [])}
    missing_model_ok = (
        row.get("preflight_blocked_missing_required_model") is True
        or "blocked_missing_required_model_preflight" in errors
    )
    return {
        **evidence_validation,
        "ok": missing_model_ok,
        "expected_block": "blocked_missing_required_model_preflight",
    }


def _run_steps(
    plan: dict[str, Any],
    *,
    refresh_blocked_preflight: bool,
) -> tuple[list[dict[str, Any]], bool]:
    runs: list[dict[str, Any]] = []
    all_ok = True
    for step in plan["steps"]:
        step_run = {
            "scenario_id": step.get("scenario_id"),
            "commands": [],
            "ok": True,
            "restore_attempted_after_failure": False,
        }
        commands = step["commands"]
        restore_command = next(
            (command for command in commands if command.get("name") == "restore"),
            None,
        )
        restore_already_run = False
        for command in commands:
            command_result = _run_shell(command["command"])
            command_result["name"] = command["name"]
            step_run["commands"].append(command_result)
            if command["name"] == "restore":
                restore_already_run = True
            if command_result["returncode"] != 0:
                if command["name"] == "run" and refresh_blocked_preflight:
                    evidence_validation = _validate_step_missing_model_preflight(step)
                    step_run["evidence_validation"] = evidence_validation
                    if evidence_validation["ok"]:
                        continue
                step_run["ok"] = False
                if (
                    restore_command
                    and not restore_already_run
                    and command["name"] != "restore"
                ):
                    restore_result = _run_shell(restore_command["command"])
                    restore_result["name"] = "restore"
                    restore_result["after_failure"] = True
                    step_run["commands"].append(restore_result)
                    step_run["restore_attempted_after_failure"] = True
                    if restore_result["returncode"] != 0:
                        step_run["restore_failed_after_failure"] = True
                break
            if command["name"] == "run":
                evidence_validation = (
                    _validate_step_missing_model_preflight(step)
                    if refresh_blocked_preflight
                    else _validate_step_result(step)
                )
                step_run["evidence_validation"] = evidence_validation
                if not evidence_validation["ok"]:
                    step_run["ok"] = False
                    if restore_command and not restore_already_run:
                        restore_result = _run_shell(restore_command["command"])
                        restore_result["name"] = "restore"
                        restore_result["after_failure"] = True
                        step_run["commands"].append(restore_result)
                        step_run["restore_attempted_after_failure"] = True
                        if restore_result["returncode"] != 0:
                            step_run["restore_failed_after_failure"] = True
                    break
        runs.append(step_run)
        if not step_run["ok"]:
            all_ok = False
            break
    return runs, all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List or execute the apron/harness closed-set candidate runtime scenarios."
    )
    parser.add_argument("--packet", default=str(DEFAULT_PRODUCTION_GATE_PACKET))
    parser.add_argument("--readiness-report", default=str(DEFAULT_READINESS_REPORT))
    parser.add_argument(
        "--out",
        help=(
            "Optional path to write the runner result JSON. Use with "
            "--refresh-blocked-preflight for durable missing-model preflight evidence."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Actually run the six scenario command blocks.")
    mode.add_argument(
        "--refresh-blocked-preflight",
        action="store_true",
        help=(
            "Run the scenario command blocks only to refresh blocked missing-model "
            "preflight JSON; fails if any scenario does not block on ppe_closed_set_candidate."
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    packet_path = Path(args.packet)
    readiness_path = Path(args.readiness_report)
    validation = validate_production_gate_packet(packet_path, readiness_report_path=readiness_path)
    packet = _load_json(packet_path)
    plan = build_plan(packet)
    mode_name = (
        "execute"
        if args.execute
        else "refresh_blocked_preflight"
        if args.refresh_blocked_preflight
        else "plan"
    )
    result: dict[str, Any] = {
        "ok": validation.get("ok") is True,
        "mode": mode_name,
        "packet": _rel(packet_path),
        "readiness_report": _rel(readiness_path),
        "packet_validation_ok": validation.get("ok") is True,
        "packet_validation_errors": validation.get("errors") or [],
        "plan": plan,
        "runs": [],
    }
    if validation.get("ok") is not True:
        result["ok"] = False
    elif args.execute and not plan["model_ready"]:
        result["ok"] = False
    elif args.refresh_blocked_preflight and plan["model_ready"]:
        result["ok"] = False
        result.setdefault("blockers", []).append("model_ready_use_execute_instead")
    elif args.execute or args.refresh_blocked_preflight:
        runs, ok = _run_steps(
            plan,
            refresh_blocked_preflight=args.refresh_blocked_preflight,
        )
        result["runs"] = runs
        result["ok"] = result["ok"] and ok

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
            "CANDIDATE_RUNTIME_RUNNER: "
            f"ok={result['ok']} mode={result['mode']} "
            f"packet_ok={result['packet_validation_ok']} "
            f"model_ready={plan['model_ready']} steps={plan['step_count']}"
        )
        for blocker in result.get("blockers", []):
            print(f"BLOCKED: {blocker}")
        for error in result["packet_validation_errors"]:
            print(f"ERROR: {error}")
        for error in plan["model_errors"]:
            print(f"BLOCKED: {error}")
        for step in plan["steps"]:
            print(f"- {step['scenario_id']} ({step['capability']}/{step['role']})")
            for command in step["commands"]:
                print(f"  {command['name']}: {command['command']}")
    if result["ok"]:
        return 0
    return 2 if args.execute and not plan["model_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
