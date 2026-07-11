#!/usr/bin/env python3
"""Plan or run guarded apron/harness closed-set candidate training steps."""

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
PLACEHOLDER_DATASET = "/path/to/cleared/dataset.yaml"
PLACEHOLDER_SEED_IMPORT = "/path/to/filled/apron_harness_seed_import_manifest.yaml"
PLACEHOLDER_TRAINING_RESULT = "/path/to/cleared/apron_harness_yolo26n_result.json"
PLACEHOLDER_TRAINING_PLAN = "/path/to/cleared/apron_harness_yolo26n_result.plan.json"
DEFAULT_CAPTURE_MANIFEST = "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml"
DEFAULT_CAPTURE_MATRIX = "qa/video_eval/results/apron_harness_production_capture_matrix.csv"
DEFAULT_CANDIDATE_REPORT = "qa/video_eval/results/apron_harness_candidate_report.json"
DEFAULT_APRON_PROMOTION_REPORT = "qa/video_eval/results/apron_closed_set_promotion_report.json"
DEFAULT_HARNESS_PROMOTION_REPORT = "qa/video_eval/results/harness_closed_set_promotion_report.json"


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


def _resolve_path(path: str | None) -> Path | None:
    if _is_placeholder(path):
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _json_status(path: str | None) -> str | None:
    candidate = _resolve_path(path)
    if not candidate or not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = payload.get("status")
    return str(status) if status is not None else None


def _load_json_file(path: str | Path) -> dict[str, Any]:
    candidate = _resolve_path(str(path))
    if not candidate or not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: str | Path) -> str | None:
    candidate = _resolve_path(str(path))
    if not candidate or not candidate.exists():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_selected_export_sha256(path: str | Path) -> str | None:
    payload = _load_json_file(path)
    manifest = payload.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return None
    handoff = manifest.get("runtime_handoff")
    if not isinstance(handoff, dict):
        return None
    selected_export = handoff.get("selected_export")
    if not isinstance(selected_export, dict):
        return None
    sha256 = str(selected_export.get("sha256") or "").strip()
    return sha256 if len(sha256) == 64 else None


def _valid_sha(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _candidate_seed_export_import_manifest(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    seed_export = manifest.get("seed_export_import_manifest")
    if isinstance(seed_export, dict):
        return seed_export
    capture_preflight = manifest.get("capture_preflight")
    if isinstance(capture_preflight, dict) and isinstance(
        capture_preflight.get("seed_export_import_manifest"), dict
    ):
        return capture_preflight["seed_export_import_manifest"]
    source_lineage = manifest.get("source_lineage")
    if isinstance(source_lineage, dict) and isinstance(
        source_lineage.get("seed_export_import_manifest"), dict
    ):
        return source_lineage["seed_export_import_manifest"]
    return {}


def _candidate_seed_lineage_ready(candidate_report: dict[str, Any]) -> tuple[bool, str]:
    seed_export = _candidate_seed_export_import_manifest(candidate_report)
    if not seed_export:
        return False, "candidate report seed_export_import_manifest is required"
    if seed_export.get("valid") is not True:
        return False, "candidate report seed_export_import_manifest.valid must be true"
    if seed_export.get("partial_materialization") is not False:
        return False, "candidate report seed_export_import_manifest.partial_materialization must be false"
    if not _valid_sha(seed_export.get("sha256")):
        return False, "candidate report seed_export_import_manifest.sha256 is missing or invalid"
    if not _valid_sha(seed_export.get("seed_source_review_sha256")):
        return False, "candidate report seed_export_import_manifest.seed_source_review_sha256 is missing or invalid"
    source_recheck = seed_export.get("source_recheck")
    if not isinstance(source_recheck, dict) or not source_recheck:
        return False, "candidate report seed_export_import_manifest.source_recheck is required"
    if source_recheck.get("exists") is not True:
        return False, "candidate report seed_export_import_manifest.source_recheck.exists must be true"
    if not source_recheck.get("path"):
        return False, "candidate report seed_export_import_manifest.source_recheck.path is required"
    if not _valid_sha(source_recheck.get("sha256")):
        return False, "candidate report seed_export_import_manifest.source_recheck.sha256 is missing or invalid"
    if "does not approve" not in str(source_recheck.get("evidence_boundary") or ""):
        return (
            False,
            "candidate report seed_export_import_manifest.source_recheck.evidence_boundary must preserve non-approval boundary",
        )
    return True, ""


def _artifact_status(name: str, path: str | None, *, output: bool = False) -> dict[str, Any]:
    resolved = _resolve_path(path)
    observed_sha256 = _sha256_file(str(resolved)) if resolved else None
    exists = bool(observed_sha256)
    return {
        "name": name,
        "path": str(path or ""),
        "resolved_path": str(resolved) if resolved else "",
        "exists": exists,
        "sha256": observed_sha256,
        "output": output,
        "ok": exists or output,
        "blockers": [] if exists or output else ["missing"],
    }


def _pinned_artifact_status(name: str, artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {
            "name": name,
            "path": "",
            "resolved_path": "",
            "exists": False,
            "sha256": None,
            "expected_sha256": "",
            "sha_matches": False,
            "output": False,
            "ok": False,
            "blockers": ["missing"],
        }
    status = _artifact_status(name, str(artifact.get("path") or ""))
    expected_sha256 = str(artifact.get("sha256") or "")
    status["expected_sha256"] = expected_sha256
    status["sha_matches"] = bool(expected_sha256 and status.get("sha256") == expected_sha256)
    status["evidence_boundary"] = artifact.get("evidence_boundary") or ""
    if status["exists"] and not status["sha_matches"]:
        status["ok"] = False
        status["blockers"] = ["sha_mismatch"]
    return status


def _candidate_report_ready(path: str | Path) -> tuple[bool, str]:
    payload = _load_json_file(path)
    if not payload:
        return False, "candidate report JSON is missing or unreadable"
    if payload.get("ok") is not True:
        return False, "candidate report ok must be true"
    manifest = payload.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return False, "candidate report promotion_manifest is required"
    if manifest.get("candidate_status") != "ready_for_side_by_side_runtime_test":
        return False, "candidate report candidate_status must be ready_for_side_by_side_runtime_test"
    if not _candidate_selected_export_sha256(path):
        return False, "candidate report selected_export.sha256 is missing or invalid"
    seed_ready, seed_reason = _candidate_seed_lineage_ready(payload)
    if not seed_ready:
        return False, seed_reason
    return True, ""


def _promotion_report_ready(
    path: str | Path,
    *,
    capability: str,
    expected_candidate_report_sha256: str,
    expected_selected_export_sha256: str,
) -> tuple[bool, str]:
    payload = _load_json_file(path)
    if not payload:
        return False, f"{capability} side-by-side promotion report is missing or unreadable"
    if payload.get("ok") is not True:
        return False, f"{capability} side-by-side promotion report ok must be true"
    if payload.get("promotion_status") != "ready_for_runtime_registration":
        return False, f"{capability} side-by-side promotion status must be ready_for_runtime_registration"
    if payload.get("capability") != capability:
        return False, f"{capability} side-by-side promotion report capability mismatch"
    if payload.get("candidate_report_sha256") != expected_candidate_report_sha256:
        return False, f"{capability} side-by-side promotion candidate_report_sha256 mismatch"
    selected_export = payload.get("candidate_selected_export")
    if not isinstance(selected_export, dict):
        return False, f"{capability} side-by-side promotion candidate_selected_export is required"
    if selected_export.get("sha256") != expected_selected_export_sha256:
        return False, f"{capability} side-by-side promotion candidate_selected_export.sha256 mismatch"
    return True, ""


def _training_plan(packet: dict[str, Any]) -> dict[str, Any]:
    plan = packet.get("candidate_training_execution_plan")
    return plan if isinstance(plan, dict) else {}


def _steps(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in (_training_plan(packet).get("steps") or [])
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
    text: str,
    *,
    dataset_yaml: str | None,
    capture_manifest: str | None,
    seed_import_manifest: str | None,
    training_result: str | None,
    training_plan_path: str | None,
) -> str:
    if dataset_yaml:
        text = text.replace(PLACEHOLDER_DATASET, dataset_yaml)
    if capture_manifest:
        text = text.replace(DEFAULT_CAPTURE_MANIFEST, capture_manifest)
    if seed_import_manifest:
        text = text.replace(PLACEHOLDER_SEED_IMPORT, seed_import_manifest)
    if training_result:
        text = text.replace(PLACEHOLDER_TRAINING_RESULT, training_result)
    if training_plan_path:
        text = text.replace(PLACEHOLDER_TRAINING_PLAN, training_plan_path)
    return text


def build_plan(
    packet: dict[str, Any],
    *,
    dataset_yaml: str | None = None,
    capture_manifest: str | None = None,
    seed_import_manifest: str | None = None,
    training_result: str | None = None,
    training_plan_path: str | None = None,
) -> dict[str, Any]:
    training = _training_plan(packet)
    success_criteria = training.get("success_criteria")
    required_input_status = (
        training.get("required_input_status")
        if isinstance(training.get("required_input_status"), dict)
        else {}
    )
    production_matrix_status = required_input_status.get("production_capture_matrix")
    label_sidecar_status = required_input_status.get("label_review_import_sidecar")
    source_recheck_artifact = required_input_status.get("source_recheck_artifact")
    production_matrix_path = (
        production_matrix_status.get("path")
        if isinstance(production_matrix_status, dict)
        else DEFAULT_CAPTURE_MATRIX
    )
    label_sidecar_path = (
        label_sidecar_status.get("path")
        if isinstance(label_sidecar_status, dict)
        else ""
    )
    return {
        "status": training.get("status"),
        "required_model_key": training.get("required_model_key"),
        "training_model": training.get("training_model"),
        "safe_to_execute_without_reviewed_data": False,
        "required_input_status": required_input_status,
        "success_criteria": success_criteria
        if isinstance(success_criteria, (dict, list))
        else [],
        "dataset_yaml_supplied": _path_exists(dataset_yaml),
        "capture_manifest_supplied": _path_exists(capture_manifest),
        "seed_import_manifest_supplied": _path_exists(seed_import_manifest),
        "training_result_supplied": not _is_placeholder(training_result),
        "artifact_status": [
            _artifact_status("dataset_yaml", dataset_yaml),
            _artifact_status("capture_manifest", capture_manifest),
            _artifact_status("seed_import_manifest", seed_import_manifest),
            _pinned_artifact_status(
                "source_recheck_artifact",
                source_recheck_artifact if isinstance(source_recheck_artifact, dict) else None,
            ),
            _artifact_status("production_capture_matrix", production_matrix_path),
            _artifact_status("label_review_import_sidecar", label_sidecar_path),
            _artifact_status("training_plan", training_plan_path, output=True),
            _artifact_status("training_result", training_result, output=True),
            _artifact_status("candidate_report", str(DEFAULT_CANDIDATE_REPORT), output=True),
            _artifact_status("apron_promotion_report", str(DEFAULT_APRON_PROMOTION_REPORT)),
            _artifact_status("harness_promotion_report", str(DEFAULT_HARNESS_PROMOTION_REPORT)),
        ],
        "step_count": len(_steps(packet)),
        "steps": [
            {
                "step": step.get("step"),
                "id": step.get("id"),
                "command": _replace_paths(
                    str(step.get("command") or ""),
                    dataset_yaml=dataset_yaml,
                    capture_manifest=capture_manifest,
                    seed_import_manifest=seed_import_manifest,
                    training_result=training_result,
                    training_plan_path=training_plan_path,
                )
                if step.get("command")
                else None,
                "commands": step.get("commands"),
                "writes": [
                    _replace_paths(
                        str(item),
                        dataset_yaml=dataset_yaml,
                        capture_manifest=capture_manifest,
                        seed_import_manifest=seed_import_manifest,
                        training_result=training_result,
                        training_plan_path=training_plan_path,
                    )
                    for item in (step.get("writes") or [])
                ],
                "pass_signal": step.get("pass_signal"),
                "guardrail": step.get("guardrail"),
            }
            for step in _steps(packet)
        ],
    }


def _block(result: dict[str, Any], reason: str) -> None:
    result["ok"] = False
    result["blocked"] = True
    result.setdefault("blockers", []).append(reason)


def _source_recheck_artifact_blockers(plan: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for artifact in plan.get("artifact_status") or []:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("name") != "source_recheck_artifact":
            continue
        if artifact.get("ok") is True:
            continue
        name = artifact.get("name") or "artifact"
        reason = ",".join(str(item) for item in (artifact.get("blockers") or [])) or "not_ok"
        blockers.append(f"{name}:{reason}")
    return blockers


def _training_packet_gate_blockers(plan: dict[str, Any]) -> list[str]:
    required_input_status = plan.get("required_input_status")
    if not isinstance(required_input_status, dict):
        return []
    blockers: list[str] = []
    production_matrix = required_input_status.get("production_capture_matrix")
    if isinstance(production_matrix, dict) and production_matrix.get("gate_passed") is not True:
        details: list[str] = []
        for key in ("missing_labeled_examples", "unapproved_rows", "unsafe_storage_rows"):
            if key in production_matrix:
                details.append(f"{key}={production_matrix.get(key)}")
        suffix = f" ({', '.join(details)})" if details else ""
        blockers.append(f"production_capture_matrix_gate_not_passed{suffix}")
    label_sidecar = required_input_status.get("label_review_import_sidecar")
    if isinstance(label_sidecar, dict) and label_sidecar.get("valid") is not True:
        error = str(label_sidecar.get("error") or "invalid")
        blockers.append(f"label_review_import_sidecar_not_valid: {error}")
    return blockers


def execute_plan(
    packet: dict[str, Any],
    *,
    dataset_yaml: str | None,
    capture_manifest: str | None,
    seed_import_manifest: str | None,
    training_result: str | None,
    training_plan_path: str | None,
    run_training: bool,
    run_candidate_doctor: bool,
    run_registry_copy: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "blocked": False,
        "blockers": [],
        "runs": [],
    }
    for label, path in (
        ("dataset YAML", dataset_yaml),
        ("reviewed capture manifest", capture_manifest),
        ("seed import manifest", seed_import_manifest),
    ):
        if _is_placeholder(path) or not _path_exists(path):
            _block(result, f"{label} must be supplied as an existing non-placeholder path")
            return result
    if _is_placeholder(training_result):
        _block(result, "--training-result must be a non-placeholder output path")
        return result

    plan = build_plan(
        packet,
        dataset_yaml=dataset_yaml,
        capture_manifest=capture_manifest,
        seed_import_manifest=seed_import_manifest,
        training_result=training_result,
        training_plan_path=training_plan_path,
    )
    source_recheck_blockers = _source_recheck_artifact_blockers(plan)
    if source_recheck_blockers:
        _block(
            result,
            "candidate training source-recheck artifact is not ready: "
            + "; ".join(source_recheck_blockers),
        )
        return result
    packet_gate_blockers = _training_packet_gate_blockers(plan)
    if packet_gate_blockers:
        _block(
            result,
            "candidate training packet gates are not ready: "
            + "; ".join(packet_gate_blockers),
        )
        return result
    by_id = {
        str(step.get("id")): step
        for step in plan["steps"]
        if isinstance(step, dict)
    }
    preflight = by_id.get("training_preflight_plan") or {}
    preflight_command = str(preflight.get("command") or "")
    if not preflight_command:
        result["ok"] = False
        result["blockers"].append("training_preflight_plan command missing")
        return result
    if "/path/to/" in preflight_command:
        _block(result, "unreplaced placeholder remains in training preflight command")
        return result
    preflight_run = _run_shell(preflight_command)
    preflight_run["step_id"] = "training_preflight_plan"
    result["runs"].append(preflight_run)
    if preflight_run["returncode"] != 0:
        result["ok"] = False
        result["blockers"].append("training preflight failed")
        return result
    if _json_status(training_plan_path) != "ready_to_train":
        _block(
            result,
            "training preflight must write a plan JSON with status=ready_to_train before training/export",
        )
        return result

    if not run_training:
        _block(result, "actual training/export is skipped until --run-training is supplied")
        return result

    train_step = by_id.get("train_export_candidate") or {}
    train_command = str(train_step.get("command") or "")
    if "/path/to/" in train_command:
        _block(result, "unreplaced placeholder remains in training command")
        return result
    training_run = _run_shell(train_command)
    training_run["step_id"] = "train_export_candidate"
    result["runs"].append(training_run)
    if training_run["returncode"] != 0:
        result["ok"] = False
        result["blockers"].append("training/export failed")
        return result
    if _json_status(training_result) != "trained":
        _block(
            result,
            "training/export must write a result JSON with status=trained before candidate doctor",
        )
        return result

    if not run_candidate_doctor:
        _block(result, "candidate doctor is skipped until --run-candidate-doctor is supplied")
        return result
    if not _path_exists(training_result):
        _block(result, "training result JSON was not created")
        return result
    candidate_step = by_id.get("candidate_doctor_report") or {}
    candidate_command = str(candidate_step.get("command") or "")
    if "/path/to/" in candidate_command:
        _block(result, "unreplaced placeholder remains in candidate doctor command")
        return result
    candidate_run = _run_shell(candidate_command)
    candidate_run["step_id"] = "candidate_doctor_report"
    result["runs"].append(candidate_run)
    if candidate_run["returncode"] != 0:
        result["ok"] = False
        result["blockers"].append("candidate doctor failed")
        return result
    candidate_ready, candidate_reason = _candidate_report_ready(DEFAULT_CANDIDATE_REPORT)
    if not candidate_ready:
        _block(result, candidate_reason)
        return result

    if not run_registry_copy:
        _block(
            result,
            "registry copy is skipped until --run-registry-copy is supplied after side-by-side promotions pass",
        )
        return result
    if not _path_exists(DEFAULT_CANDIDATE_REPORT):
        _block(result, "candidate report JSON is missing")
        return result
    candidate_report_sha256 = _sha256_file(DEFAULT_CANDIDATE_REPORT)
    if not candidate_report_sha256:
        _block(result, "candidate report SHA could not be computed")
        return result
    selected_export_sha256 = _candidate_selected_export_sha256(DEFAULT_CANDIDATE_REPORT)
    if not selected_export_sha256:
        _block(result, "candidate report selected_export.sha256 is missing or invalid")
        return result
    for capability, promotion_path in (
        ("apron_required", DEFAULT_APRON_PROMOTION_REPORT),
        ("harness_required", DEFAULT_HARNESS_PROMOTION_REPORT),
    ):
        ready, reason = _promotion_report_ready(
            promotion_path,
            capability=capability,
            expected_candidate_report_sha256=candidate_report_sha256,
            expected_selected_export_sha256=selected_export_sha256,
        )
        if not ready:
            _block(result, reason)
            return result
    registry_step = by_id.get("registry_copy_after_promotions") or {}
    registry_command = str(registry_step.get("command") or "")
    registry_run = _run_shell(registry_command)
    registry_run["step_id"] = "registry_copy_after_promotions"
    result["runs"].append(registry_run)
    if registry_run["returncode"] != 0:
        result["ok"] = False
        result["blockers"].append("registry copy failed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute guarded apron/harness closed-set candidate training."
    )
    parser.add_argument("--packet", default=str(DEFAULT_PRODUCTION_GATE_PACKET))
    parser.add_argument("--readiness-report", default=str(DEFAULT_READINESS_REPORT))
    parser.add_argument("--dataset-yaml", default="")
    parser.add_argument("--capture-manifest", default=DEFAULT_CAPTURE_MANIFEST)
    parser.add_argument("--seed-import-manifest", default="")
    parser.add_argument("--training-result", default="")
    parser.add_argument("--training-plan", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-candidate-doctor", action="store_true")
    parser.add_argument("--run-registry-copy", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    packet_path = Path(args.packet)
    readiness_path = Path(args.readiness_report)
    validation = validate_production_gate_packet(packet_path, readiness_report_path=readiness_path)
    packet = _load_json(packet_path)
    training_plan_path = args.training_plan or (
        str(Path(args.training_result).with_suffix(".plan.json"))
        if args.training_result and not _is_placeholder(args.training_result)
        else ""
    )
    plan = build_plan(
        packet,
        dataset_yaml=args.dataset_yaml or None,
        capture_manifest=args.capture_manifest or None,
        seed_import_manifest=args.seed_import_manifest or None,
        training_result=args.training_result or None,
        training_plan_path=training_plan_path or None,
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
            dataset_yaml=args.dataset_yaml or None,
            capture_manifest=args.capture_manifest or None,
            seed_import_manifest=args.seed_import_manifest or None,
            training_result=args.training_result or None,
            training_plan_path=training_plan_path or None,
            run_training=bool(args.run_training),
            run_candidate_doctor=bool(args.run_candidate_doctor),
            run_registry_copy=bool(args.run_registry_copy),
        )
        result.update(execution)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            "CANDIDATE_TRAINING_RUNNER: "
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
