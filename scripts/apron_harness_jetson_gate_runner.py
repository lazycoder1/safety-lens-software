#!/usr/bin/env python3
"""Plan or run the guarded apron/harness Jetson full gate."""

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


DEFAULT_READINESS_REPORT = DEFAULT_RESULT_DIR / "apron_harness_readiness_doctor.json"
DEFAULT_OUT = DEFAULT_RESULT_DIR / "factory_ppe_jetson_gate.json"
PLACEHOLDER_RAW = "/path/to/cleared/factory_ppe_raw_benchmark.json"
PLACEHOLDER_SOAK = "/path/to/cleared/factory_ppe_3cam_soak.json"
DEFAULT_CANDIDATE_REPORT = "qa/video_eval/results/apron_harness_candidate_report.json"


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return "/path/to/" in value or value.startswith("/path/to")


def _artifact(path: str | None) -> Path:
    candidate = Path(str(path or ""))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _path_exists(path: str | None) -> bool:
    return not _is_placeholder(path) and _artifact(path).exists()


def _load_gate_output(path: str | None) -> dict[str, Any]:
    if _is_placeholder(path):
        return {}
    candidate = _artifact(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_status(name: str, path: str | None, *, output: bool = False) -> dict[str, Any]:
    if _is_placeholder(path):
        return {
            "name": name,
            "path": str(path or ""),
            "exists": False,
            "sha256": None,
            "output": output,
            "ok": output,
            "blockers": [] if output else ["missing"],
        }
    candidate = _artifact(path)
    exists = candidate.exists() and candidate.is_file()
    return {
        "name": name,
        "path": str(path or ""),
        "exists": exists,
        "sha256": _sha256_file(candidate) if exists else None,
        "output": output,
        "ok": exists or output,
        "blockers": [] if exists or output else ["missing"],
    }


def _same_artifact(observed: str | None, expected: str | None) -> bool:
    if not observed or not expected:
        return False
    return _artifact(observed).resolve() == _artifact(expected).resolve()


def _gate_output_matches_inputs(
    gate_output: dict[str, Any],
    *,
    candidate_report: str,
    raw_benchmark: str,
    soak_report: str,
) -> list[str]:
    blockers: list[str] = []
    inputs = gate_output.get("inputs")
    if not isinstance(inputs, dict):
        return ["Jetson full gate output must include input path identity"]
    for key, expected in (
        ("candidate_report", candidate_report),
        ("raw_benchmark", raw_benchmark),
        ("soak_report", soak_report),
    ):
        if not _same_artifact(inputs.get(key), expected):
            blockers.append(f"Jetson full gate output input mismatch: {key}")

    expected_shas = {
        "candidate_report": _sha256_file(_artifact(candidate_report)),
        "raw_benchmark": _sha256_file(_artifact(raw_benchmark)),
        "soak_report": _sha256_file(_artifact(soak_report)),
    }
    observed_shas = gate_output.get("input_file_sha256s")
    if not isinstance(observed_shas, dict):
        blockers.append("Jetson full gate output must include input_file_sha256s")
        return blockers
    for key, expected_sha in expected_shas.items():
        if observed_shas.get(key) != expected_sha:
            blockers.append(f"Jetson full gate output file hash mismatch: {key}")
    return blockers


def _jetson_plan(packet: dict[str, Any]) -> dict[str, Any]:
    plan = packet.get("jetson_gate_execution_plan")
    return plan if isinstance(plan, dict) else {}


def _gate_action(packet: dict[str, Any]) -> dict[str, Any]:
    for action in packet.get("next_actions") or []:
        if isinstance(action, dict) and action.get("id") == "prove_edge_gate":
            return action
    return {}


def _replace_inputs(
    text: str,
    *,
    candidate_report: str | None,
    raw_benchmark: str | None,
    soak_report: str | None,
    out_path: str | None,
) -> str:
    if candidate_report:
        text = text.replace(DEFAULT_CANDIDATE_REPORT, candidate_report)
    if raw_benchmark:
        text = text.replace(PLACEHOLDER_RAW, raw_benchmark)
    if soak_report:
        text = text.replace(PLACEHOLDER_SOAK, soak_report)
    if out_path:
        text = text.replace("qa/video_eval/results/factory_ppe_jetson_gate.json", out_path)
    return text


def build_plan(
    packet: dict[str, Any],
    *,
    candidate_report: str | None = None,
    raw_benchmark: str | None = None,
    soak_report: str | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    jetson_plan = _jetson_plan(packet)
    gate_action = _gate_action(packet)
    command = str((gate_action.get("command") or jetson_plan.get("full_gate_command") or ""))
    rendered_command = _replace_inputs(
        command,
        candidate_report=candidate_report,
        raw_benchmark=raw_benchmark,
        soak_report=soak_report,
        out_path=out_path,
    )
    templates = jetson_plan.get("templates") or {}
    if not isinstance(templates, dict):
        templates = {}
    return {
        "status": jetson_plan.get("status"),
        "required_model_key": jetson_plan.get("required_model_key"),
        "model": jetson_plan.get("model"),
        "safe_to_execute_without_candidate_evidence": False,
        "candidate_report_supplied": _path_exists(candidate_report),
        "raw_benchmark_supplied": _path_exists(raw_benchmark),
        "soak_report_supplied": _path_exists(soak_report),
        "artifact_status": [
            _artifact_status("candidate_report", candidate_report),
            _artifact_status("raw_benchmark", raw_benchmark),
            _artifact_status("soak_report", soak_report),
            _artifact_status("gate_output", out_path, output=True),
        ],
        "templates": templates,
        "full_gate": jetson_plan.get("full_gate") or packet.get("jetson_full_gate") or {},
        "full_gate_command": _replace_inputs(
            str(jetson_plan.get("full_gate_command") or ""),
            candidate_report=candidate_report,
            raw_benchmark=raw_benchmark,
            soak_report=soak_report,
            out_path=out_path,
        ),
        "success_criteria": jetson_plan.get("success_criteria")
        if isinstance(jetson_plan.get("success_criteria"), list)
        else [],
        "required_inputs": [
            "candidate_report",
            "raw_benchmark",
            "soak_report",
        ],
        "step_count": len(jetson_plan.get("steps") or []),
        "steps": jetson_plan.get("steps") or [],
        "command": rendered_command,
    }


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


def _block(result: dict[str, Any], reason: str) -> None:
    result["ok"] = False
    result["blocked"] = True
    result.setdefault("blockers", []).append(reason)


def execute_plan(
    packet: dict[str, Any],
    *,
    candidate_report: str | None,
    raw_benchmark: str | None,
    soak_report: str | None,
    out_path: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "blocked": False,
        "blockers": [],
        "runs": [],
    }
    for label, path in (
        ("candidate report", candidate_report),
        ("raw benchmark", raw_benchmark),
        ("soak report", soak_report),
    ):
        if not _path_exists(path):
            _block(result, f"{label} must be supplied as an existing non-placeholder path")
            return result

    plan = build_plan(
        packet,
        candidate_report=candidate_report,
        raw_benchmark=raw_benchmark,
        soak_report=soak_report,
        out_path=out_path,
    )
    command = str(plan.get("command") or "")
    if not command:
        result["ok"] = False
        result["blockers"].append("full Jetson gate command missing")
        return result
    if "/path/to/" in command:
        _block(result, "unreplaced placeholder remains in full Jetson gate command")
        return result
    if "--require-full-gate" not in command:
        result["ok"] = False
        result["blockers"].append("full Jetson gate command must include --require-full-gate")
        return result

    gate_run = _run_shell(command)
    gate_run["step_id"] = "validate_full_gate"
    result["runs"].append(gate_run)
    if gate_run["returncode"] != 0:
        result["ok"] = False
        result["blockers"].append("Jetson full gate failed")
        return result
    gate_output = _load_gate_output(out_path)
    result["gate_output"] = {
        "path": str(_artifact(out_path)),
        "present": bool(gate_output),
        "gate_status": gate_output.get("gate_status"),
        "ok": gate_output.get("ok"),
    }
    if gate_output.get("gate_status") != "jetson_gate_passed":
        _block(result, "Jetson full gate output must report gate_status=jetson_gate_passed")
        return result
    if gate_output.get("ok") is not True:
        _block(result, "Jetson full gate output must report ok=true")
    input_blockers = _gate_output_matches_inputs(
        gate_output,
        candidate_report=candidate_report,
        raw_benchmark=raw_benchmark,
        soak_report=soak_report,
    )
    for blocker in input_blockers:
        _block(result, blocker)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute guarded apron/harness Jetson full gate.")
    parser.add_argument("--packet", default=str(DEFAULT_PRODUCTION_GATE_PACKET))
    parser.add_argument("--readiness-report", default=str(DEFAULT_READINESS_REPORT))
    parser.add_argument("--candidate-report", default="")
    parser.add_argument("--raw-benchmark", default="")
    parser.add_argument("--soak-report", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    packet_path = Path(args.packet)
    readiness_path = Path(args.readiness_report)
    validation = validate_production_gate_packet(packet_path, readiness_report_path=readiness_path)
    packet = _load_json(packet_path)
    plan = build_plan(
        packet,
        candidate_report=args.candidate_report or None,
        raw_benchmark=args.raw_benchmark or None,
        soak_report=args.soak_report or None,
        out_path=args.out or None,
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
            candidate_report=args.candidate_report or None,
            raw_benchmark=args.raw_benchmark or None,
            soak_report=args.soak_report or None,
            out_path=args.out or None,
        )
        result.update(execution)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            "JETSON_GATE_RUNNER: "
            f"ok={result['ok']} mode={result['mode']} "
            f"packet_ok={result['packet_validation_ok']} "
            f"steps={plan['step_count']} blocked={result['blocked']}"
        )
        for error in result["packet_validation_errors"]:
            print(f"ERROR: {error}")
        for blocker in result["blockers"]:
            print(f"BLOCKED: {blocker}")
        if plan.get("command"):
            print(f"command: {plan['command']}")
    if result["ok"]:
        return 0
    if result.get("blocked"):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
