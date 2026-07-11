#!/usr/bin/env python3
"""Validate Jetson benchmark evidence against model-pack resource limits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PACKS = ROOT / "qa" / "video_eval" / "model_packs.yaml"
FACTORY_PPE_PACK_ID = "factory_ppe_3cam"
REQUIRED_FACTORY_PPE_CLASSES = {
    "person",
    "apron",
    "safety_harness",
    "safety_lanyard",
}
VISIBLE_PPE_CLASS_HINTS = {
    "apron_required": {
        "apron",
        "protective apron",
        "kitchen apron",
        "denim apron",
        "work apron",
    },
    "harness_required": {
        "safety harness",
        "fall arrest harness",
        "body harness",
        "harness",
        "safety lanyard",
        "fall protection lanyard",
        "safety_harness",
        "safety_lanyard",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pack(pack_path: Path, pack_id: str) -> dict[str, Any]:
    doc = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
    packs = doc.get("packs") or {}
    pack = packs.get(pack_id)
    if not isinstance(pack, dict):
        raise KeyError(f"unknown model pack: {pack_id}")
    return pack


def _basename(value: str) -> str:
    return Path(str(value)).name


def _model_matches(observed: str, expected: str) -> bool:
    return str(observed) == str(expected) or _basename(observed) == _basename(expected)


def _sha_from_container(container: Any) -> str | None:
    if not isinstance(container, dict):
        return None
    for key in (
        "model_artifact_sha256",
        "artifact_sha256",
        "selected_export_sha256",
        "source_export_sha256",
    ):
        value = container.get(key)
        if value:
            return str(value)
    for key in ("model_artifact", "artifact", "selected_export"):
        nested = container.get(key)
        if isinstance(nested, dict) and nested.get("sha256"):
            return str(nested["sha256"])
    return None


def _valid_sha(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _candidate_report_sha_from_container(container: Any) -> str | None:
    if not isinstance(container, dict):
        return None
    value = container.get("candidate_report_sha256")
    if value:
        return str(value)
    nested = container.get("candidate_report")
    if isinstance(nested, dict) and nested.get("sha256"):
        return str(nested["sha256"])
    return None


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


def _validate_source_recheck_block(source_recheck: Any, *, prefix: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(source_recheck, dict) or not source_recheck:
        errors.append(f"{prefix}.source_recheck is required")
        return {}
    if source_recheck.get("exists") is not True:
        errors.append(f"{prefix}.source_recheck.exists must be true")
    if not source_recheck.get("path"):
        errors.append(f"{prefix}.source_recheck.path is required")
    if not _valid_sha(source_recheck.get("sha256")):
        errors.append(f"{prefix}.source_recheck.sha256 must be a 64-character digest")
    if "does not approve" not in str(source_recheck.get("evidence_boundary") or ""):
        errors.append(f"{prefix}.source_recheck.evidence_boundary must preserve non-approval boundary")
    return source_recheck


def _candidate_report_identity(
    path: Path,
    errors: list[str],
    *,
    require_seed_source_recheck: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "present": False,
        "path": str(path),
        "sha256": None,
        "ok": None,
        "candidate_status": None,
        "selected_export_sha256": None,
        "selected_export_path": None,
        "seed_export_import_manifest": None,
        "seed_source_recheck": None,
    }
    try:
        summary["sha256"] = _sha256_file(path)
        payload = _load_json(path)
    except FileNotFoundError:
        errors.append(f"candidate report missing: {path}")
        return summary
    except json.JSONDecodeError as exc:
        errors.append(f"candidate report is not valid JSON: {path}: {exc}")
        return summary
    except OSError as exc:
        errors.append(f"candidate report unavailable: {path}: {exc}")
        return summary

    summary["present"] = True
    summary["ok"] = payload.get("ok")
    manifest = payload.get("promotion_manifest")
    if not isinstance(manifest, dict):
        errors.append("candidate report missing promotion_manifest")
        return summary

    summary["candidate_status"] = manifest.get("candidate_status")
    metric_thresholds = manifest.get("metric_thresholds") if isinstance(manifest.get("metric_thresholds"), dict) else {}
    class_metrics = manifest.get("class_metrics") if isinstance(manifest.get("class_metrics"), dict) else {}
    min_map50 = _metric_float(metric_thresholds, "min_per_class_mAP50")
    min_recall = _metric_float(metric_thresholds, "min_per_class_recall")
    if min_map50 is None:
        errors.append("candidate report missing promotion_manifest.metric_thresholds.min_per_class_mAP50")
    if min_recall is None:
        errors.append("candidate report missing promotion_manifest.metric_thresholds.min_per_class_recall")
    summary["metric_thresholds"] = {
        "min_per_class_mAP50": min_map50,
        "min_per_class_recall": min_recall,
    }
    metric_summary: dict[str, dict[str, float | None]] = {}
    for class_name in sorted(REQUIRED_FACTORY_PPE_CLASSES):
        metrics = class_metrics.get(class_name)
        if not isinstance(metrics, dict):
            errors.append(f"candidate report missing promotion_manifest.class_metrics.{class_name}")
            metric_summary[class_name] = {"mAP50": None, "recall": None}
            continue
        map50 = _metric_float(metrics, "mAP50")
        recall = _metric_float(metrics, "recall")
        metric_summary[class_name] = {"mAP50": map50, "recall": recall}
        if map50 is None:
            errors.append(f"candidate report missing promotion_manifest.class_metrics.{class_name}.mAP50")
        elif min_map50 is not None and map50 < min_map50:
            errors.append(f"candidate report promotion_manifest.class_metrics.{class_name}.mAP50 below threshold")
        if recall is None:
            errors.append(f"candidate report missing promotion_manifest.class_metrics.{class_name}.recall")
        elif min_recall is not None and recall < min_recall:
            errors.append(f"candidate report promotion_manifest.class_metrics.{class_name}.recall below threshold")
    summary["class_metrics"] = metric_summary
    handoff = manifest.get("runtime_handoff")
    selected_export = handoff.get("selected_export") if isinstance(handoff, dict) else None
    if isinstance(selected_export, dict):
        summary["selected_export_sha256"] = selected_export.get("sha256")
        summary["selected_export_path"] = selected_export.get("path")
    seed_export = _candidate_seed_export_import_manifest(payload)
    if seed_export:
        summary["seed_export_import_manifest"] = {
            "required": seed_export.get("required"),
            "valid": seed_export.get("valid"),
            "partial_materialization": seed_export.get("partial_materialization"),
            "sha256": seed_export.get("sha256"),
            "seed_source_review_sha256": seed_export.get("seed_source_review_sha256"),
            "source_recheck": seed_export.get("source_recheck"),
        }
        summary["seed_source_recheck"] = seed_export.get("source_recheck")
    if require_seed_source_recheck:
        if not seed_export:
            errors.append("candidate report must include seed_export_import_manifest")
        else:
            if seed_export.get("valid") is not True:
                errors.append("candidate seed_export_import_manifest.valid must be true")
            if seed_export.get("partial_materialization") is not False:
                errors.append("candidate seed_export_import_manifest.partial_materialization must be false")
            if not _valid_sha(seed_export.get("sha256")):
                errors.append("candidate seed_export_import_manifest.sha256 must be a 64-character digest")
            if not _valid_sha(seed_export.get("seed_source_review_sha256")):
                errors.append(
                    "candidate seed_export_import_manifest.seed_source_review_sha256 must be a 64-character digest"
                )
            _validate_source_recheck_block(
                seed_export.get("source_recheck"),
                prefix="candidate seed_export_import_manifest",
                errors=errors,
            )

    if payload.get("ok") is not True:
        errors.append("candidate report must have ok=true")
    if manifest.get("candidate_status") != "ready_for_side_by_side_runtime_test":
        errors.append(
            "candidate report promotion_manifest.candidate_status must be "
            "ready_for_side_by_side_runtime_test"
        )
    if not summary["selected_export_sha256"]:
        errors.append("candidate report missing promotion_manifest.runtime_handoff.selected_export.sha256")
    return summary


def _metric_float(container: Any, key: str) -> float | None:
    if not isinstance(container, dict):
        return None
    try:
        value = container.get(key)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(report: dict[str, Any], name: str) -> Any:
    if name in report:
        return report[name]
    for container_name in ("metrics", "resource_metrics", "telemetry"):
        container = report.get(container_name)
        if isinstance(container, dict) and name in container:
            return container[name]
    return None


def _float_field(entry: dict[str, Any], key: str, errors: list[str], *, label: str) -> float | None:
    value = entry.get(key)
    if value is None:
        errors.append(f"{label} missing {key}")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} {key} must be numeric")
        return None


def _int_field(entry: dict[str, Any], key: str, errors: list[str], *, label: str) -> int | None:
    value = entry.get(key)
    if value is None:
        errors.append(f"{label} missing {key}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} {key} must be an integer")
        return None


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, list):
        values = value
    else:
        values = [value]

    numbers: list[float] = []
    for item in values:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return numbers


def _numeric_metric(
    report: dict[str, Any],
    name: str,
    errors: list[str],
    *,
    missing_ok: bool = False,
    use_max: bool = False,
) -> float | None:
    value = _metric(report, name)
    if value is None:
        if not missing_ok:
            errors.append(f"soak report missing metric: {name}")
        return None
    values = _numeric_values(value)
    if not values:
        errors.append(f"soak report metric must be numeric: {name}")
        return None
    return max(values) if use_max else min(values)


def _fps_values(value: Any) -> list[float]:
    return _numeric_values(value)


def _camera_count(report: dict[str, Any]) -> int | None:
    for name in ("camera_count", "max_cameras", "active_cameras"):
        value = _metric(report, name)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    cameras = _metric(report, "cameras")
    if isinstance(cameras, list):
        return len(cameras)
    try:
        if cameras is not None:
            return int(cameras)
    except (TypeError, ValueError):
        pass
    return None


def _p95_limit(limits: dict[str, Any]) -> float:
    explicit = limits.get("max_p95_latency_ms") or limits.get("max_p95_latency_ms_per_frame")
    if explicit is not None:
        return float(explicit)
    mean_limits = [
        float(limits.get("max_mean_latency_ms_per_frame") or 0),
        float(limits.get("max_model_server_mean_latency_ms_per_request") or 0),
    ]
    return max(mean_limits) * 1.5


def _find_raw_model(raw: dict[str, Any], model: str | None, errors: list[str]) -> dict[str, Any] | None:
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        errors.append("raw benchmark must include a non-empty models list")
        return None

    if model:
        for entry in models:
            if isinstance(entry, dict) and _model_matches(str(entry.get("model") or ""), model):
                return entry
        errors.append(f"raw benchmark does not include model: {model}")
        return None

    dict_models = [entry for entry in models if isinstance(entry, dict)]
    if len(dict_models) == 1:
        return dict_models[0]
    errors.append("--model is required when raw benchmark contains multiple models")
    return None


def _validate_raw_benchmark(
    raw: dict[str, Any],
    limits: dict[str, Any],
    *,
    model: str | None,
    allow_cpu: bool,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "present": True,
        "cuda": bool(raw.get("cuda")),
        "device": raw.get("device"),
        "model_artifact_sha256": _sha_from_container(raw),
        "candidate_report_sha256": _candidate_report_sha_from_container(raw),
    }
    if not allow_cpu and raw.get("cuda") is not True:
        errors.append("raw benchmark must run with cuda=true on Jetson")

    max_cameras = int(limits.get("max_cameras") or 1)
    frames = raw.get("frames")
    frame_count = len(frames) if isinstance(frames, list) else 0
    summary["frame_count"] = frame_count
    if frame_count < max_cameras:
        errors.append(f"raw benchmark must include at least {max_cameras} camera frames")

    entry = _find_raw_model(raw, model, errors)
    if entry is None:
        return summary

    mean_ms = _float_field(entry, "mean_ms", errors, label="raw benchmark")
    p95_ms = _float_field(entry, "p95_ms", errors, label="raw benchmark")
    explicit_single_stream_fps = entry.get("fps_single_stream_estimate")
    if explicit_single_stream_fps is None:
        single_stream_fps = 1000.0 / mean_ms if mean_ms else 0
    else:
        try:
            single_stream_fps = float(explicit_single_stream_fps)
        except (TypeError, ValueError):
            errors.append("raw benchmark fps_single_stream_estimate must be numeric")
            single_stream_fps = 0
    estimated_fps_per_camera = single_stream_fps / max_cameras if max_cameras else 0
    samples = _int_field(entry, "samples", errors, label="raw benchmark")

    summary.update(
        {
            "model": entry.get("model"),
            "model_artifact_sha256": _sha_from_container(entry) or summary.get("model_artifact_sha256"),
            "candidate_report_sha256": _candidate_report_sha_from_container(entry)
            or summary.get("candidate_report_sha256"),
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "samples": samples or 0,
            "fps_single_stream_estimate": single_stream_fps,
            "estimated_fps_per_camera_at_max_cameras": round(estimated_fps_per_camera, 3),
            "minimum_acceptable_fps_per_camera": limits.get("minimum_acceptable_fps_per_camera"),
        }
    )

    if samples is not None and samples < max(frame_count, max_cameras):
        errors.append("raw benchmark has too few latency samples")
    if mean_ms is None:
        pass
    elif mean_ms <= 0:
        errors.append("raw benchmark mean_ms must be positive")
    elif mean_ms > float(limits["max_mean_latency_ms_per_frame"]):
        errors.append(
            f"raw benchmark mean_ms {mean_ms} exceeds limit "
            f"{limits['max_mean_latency_ms_per_frame']}"
        )
    if p95_ms is None:
        pass
    elif p95_ms <= 0:
        errors.append("raw benchmark p95_ms must be positive")
    elif p95_ms > _p95_limit(limits):
        errors.append(f"raw benchmark p95_ms {p95_ms} exceeds derived limit {_p95_limit(limits)}")
    if estimated_fps_per_camera < float(limits["minimum_acceptable_fps_per_camera"]):
        errors.append(
            f"raw benchmark estimated fps/camera {estimated_fps_per_camera:.2f} is below "
            f"{limits['minimum_acceptable_fps_per_camera']}"
        )
    elif estimated_fps_per_camera < float(limits["target_fps_per_camera"]):
        warnings.append(
            f"raw benchmark estimated fps/camera {estimated_fps_per_camera:.2f} is below target "
            f"{limits['target_fps_per_camera']} but above minimum"
        )
    return summary


def _validate_required_metrics(report: dict[str, Any], limits: dict[str, Any], errors: list[str]) -> list[str]:
    missing = []
    for metric_name in limits.get("required_metrics") or []:
        if _metric(report, str(metric_name)) is None:
            missing.append(str(metric_name))
    if missing:
        errors.append(f"soak report missing required metrics: {', '.join(missing)}")
    return missing


def _validate_required_alert_capabilities(
    per_class_alerts: Any,
    limits: dict[str, Any],
    errors: list[str],
) -> dict[str, float]:
    required = [str(value) for value in limits.get("required_positive_alert_capabilities") or []]
    if not required:
        return {}
    if not isinstance(per_class_alerts, dict):
        errors.append(
            "soak report per_class_alert_count must include required capabilities: "
            f"{', '.join(required)}"
        )
        return {}

    summary: dict[str, float] = {}
    missing_or_zero: list[str] = []
    for capability in required:
        raw_value = per_class_alerts.get(capability)
        try:
            count = float(raw_value)
        except (TypeError, ValueError):
            count = 0.0
        summary[capability] = count
        if count <= 0:
            missing_or_zero.append(capability)
    if missing_or_zero:
        errors.append(
            "soak report per_class_alert_count must include positive counts for: "
            f"{', '.join(missing_or_zero)}"
        )
    return summary


def _suppression_metric(entry: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = entry.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _suppression_model_invocations(entry: dict[str, Any]) -> dict[str, float]:
    raw = entry.get("model_invocations")
    if raw is None:
        raw = entry.get("modelInvocationCounts")
    if raw is None:
        raw = entry.get("model_invocation_counts")
    if not isinstance(raw, dict):
        return {}
    invocations: dict[str, float] = {}
    for model_key, value in raw.items():
        try:
            invocations[str(model_key)] = float(value)
        except (TypeError, ValueError):
            invocations[str(model_key)] = 1.0
    return invocations


def _validate_detector_window_suppression(
    suppression: Any,
    limits: dict[str, Any],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    required = [str(value) for value in limits.get("required_detector_window_suppression_capabilities") or []]
    if not required:
        return {}
    if not isinstance(suppression, dict):
        errors.append(
            "soak report detector_window_suppression must include required capabilities: "
            f"{', '.join(required)}"
        )
        return {}

    summary: dict[str, dict[str, Any]] = {}
    for capability in required:
        entry = suppression.get(capability)
        if not isinstance(entry, dict):
            errors.append(f"soak report detector_window_suppression missing capability: {capability}")
            continue
        suppressed = entry.get("suppressed") is True or capability in {
            str(value) for value in (entry.get("suppressed_capabilities") or [])
        }
        max_detections = _suppression_metric(entry, "max_detections", "max_detections_count", "detections", "candidate_count")
        matching_alerts = _suppression_metric(entry, "matching_alerts", "alert_count", "alerts")
        unexpected_alerts = _suppression_metric(entry, "unexpected_alerts", "forbidden_alerts")
        invocations = _suppression_model_invocations(entry)
        summary[capability] = {
            "suppressed": suppressed,
            "max_detections": max_detections,
            "matching_alerts": matching_alerts,
            "unexpected_alerts": unexpected_alerts,
            "model_invocations": invocations,
        }
        if not suppressed:
            errors.append(f"soak report detector_window_suppression must mark {capability} as suppressed")
        if max_detections is None:
            errors.append(f"soak report detector_window_suppression missing max_detections for {capability}")
        elif max_detections != 0:
            errors.append(f"soak report detector_window_suppression {capability} emitted detections")
        if matching_alerts is None:
            errors.append(f"soak report detector_window_suppression missing alert count for {capability}")
        elif matching_alerts != 0:
            errors.append(f"soak report detector_window_suppression {capability} emitted alerts")
        if unexpected_alerts is not None and unexpected_alerts != 0:
            errors.append(f"soak report detector_window_suppression {capability} emitted unexpected alerts")
        if not invocations:
            errors.append(f"soak report detector_window_suppression missing model_invocations for {capability}")
        elif any(value != 0 for value in invocations.values()):
            errors.append(f"soak report detector_window_suppression {capability} invoked a model")
    return summary


def _validate_false_positive_guards(
    guard: Any,
    limits: dict[str, Any],
    errors: list[str],
) -> dict[str, dict[str, float | None]]:
    required = [str(value) for value in limits.get("required_false_positive_guard_capabilities") or []]
    if not required:
        return {}
    if not isinstance(guard, dict):
        errors.append(
            "soak report false_positive_guard must include required capabilities: "
            f"{', '.join(required)}"
        )
        return {}

    summary: dict[str, dict[str, float | None]] = {}
    for capability in required:
        entry = guard.get(capability)
        if not isinstance(entry, dict):
            errors.append(f"soak report false_positive_guard missing capability: {capability}")
            continue
        visible_class_total = _suppression_metric(
            entry,
            "visible_class_total",
            "visible_positive_class_total",
            "visible_ppe_class_total",
        )
        matching_alerts = _suppression_metric(entry, "matching_alerts", "alert_count", "alerts")
        unexpected_alerts = _suppression_metric(entry, "unexpected_alerts", "forbidden_alerts")
        false_positive_count = _suppression_metric(entry, "false_positive_count")
        summary[capability] = {
            "visible_class_total": visible_class_total,
            "matching_alerts": matching_alerts,
            "unexpected_alerts": unexpected_alerts,
            "false_positive_count": false_positive_count,
        }
        if visible_class_total is None:
            errors.append(f"soak report false_positive_guard missing visible_class_total for {capability}")
        elif visible_class_total <= 0:
            errors.append(f"soak report false_positive_guard {capability} must include visible PPE evidence")
        if matching_alerts is None:
            errors.append(f"soak report false_positive_guard missing alert count for {capability}")
        elif matching_alerts != 0:
            errors.append(f"soak report false_positive_guard {capability} emitted alerts")
        if unexpected_alerts is not None and unexpected_alerts != 0:
            errors.append(f"soak report false_positive_guard {capability} emitted unexpected alerts")
        if false_positive_count is not None and false_positive_count != 0:
            errors.append(f"soak report false_positive_guard {capability} has false positives")
    return summary


def _validate_soak_report(
    soak: dict[str, Any],
    limits: dict[str, Any],
    *,
    pack_id: str,
    model: str | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "present": True,
        "model_artifact_sha256": _sha_from_container(soak),
        "candidate_report_sha256": _candidate_report_sha_from_container(soak),
    }
    if soak.get("pack_id") and soak.get("pack_id") != pack_id:
        errors.append(f"soak report pack_id must be {pack_id}")
    if model and soak.get("model") and not _model_matches(str(soak["model"]), model):
        errors.append(f"soak report model must match {model}")
    source_failures = soak.get("source_result_failures")
    if isinstance(source_failures, list) and source_failures:
        errors.append(
            "soak report source results must be passing before they can be used as evidence: "
            f"{', '.join(str(value) for value in source_failures)}"
        )

    max_cameras = int(limits["max_cameras"])
    camera_count = _camera_count(soak)
    summary["camera_count"] = camera_count
    if camera_count is None:
        errors.append("soak report must include camera_count, active_cameras, max_cameras, or cameras")
    elif camera_count < max_cameras:
        errors.append(f"soak report must cover {max_cameras} cameras")

    _validate_required_metrics(soak, limits, errors)

    soak_minutes = _numeric_metric(soak, "soak_minutes", errors)
    fps_raw = _metric(soak, "fps_per_camera")
    fps_values = _fps_values(fps_raw)
    if not fps_values:
        errors.append("soak report fps_per_camera must include numeric values")
        min_fps = None
    else:
        min_fps = min(fps_values)
        if isinstance(fps_raw, (list, dict)) and len(fps_values) < max_cameras:
            errors.append(f"soak report fps_per_camera must include {max_cameras} camera values")

    mean_latency = _numeric_metric(soak, "mean_latency_ms", errors)
    p95_latency = _numeric_metric(soak, "p95_latency_ms", errors)
    model_server_mean = _numeric_metric(soak, "model_server_mean_latency_ms_per_request", errors)
    ram_mb = _numeric_metric(soak, "ram_mb", errors, use_max=True)
    gpu_utilization = _numeric_metric(soak, "gpu_utilization_percent", errors, use_max=True)
    stream_restarts = _numeric_metric(soak, "stream_restarts", errors, use_max=True)
    false_positive_count = _numeric_metric(soak, "false_positive_count", errors, missing_ok=True, use_max=True)

    summary.update(
        {
            "soak_minutes": soak_minutes,
            "min_fps_per_camera": min_fps,
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "model_server_mean_latency_ms_per_request": model_server_mean,
            "ram_mb": ram_mb,
            "gpu_utilization_percent": gpu_utilization,
            "stream_restarts": stream_restarts,
            "false_positive_count": false_positive_count,
        }
    )

    if soak_minutes is not None and soak_minutes < float(limits["soak_minutes"]):
        errors.append(f"soak_minutes {soak_minutes} is below required {limits['soak_minutes']}")
    if min_fps is not None:
        if min_fps < float(limits["minimum_acceptable_fps_per_camera"]):
            errors.append(
                f"min fps_per_camera {min_fps} is below minimum "
                f"{limits['minimum_acceptable_fps_per_camera']}"
            )
        elif min_fps < float(limits["target_fps_per_camera"]):
            warnings.append(
                f"min fps_per_camera {min_fps} is below target {limits['target_fps_per_camera']} "
                "but above minimum"
            )
    if mean_latency is not None and mean_latency > float(limits["max_mean_latency_ms_per_frame"]):
        errors.append(
            f"mean_latency_ms {mean_latency} exceeds limit {limits['max_mean_latency_ms_per_frame']}"
        )
    if p95_latency is not None and p95_latency > _p95_limit(limits):
        errors.append(f"p95_latency_ms {p95_latency} exceeds derived limit {_p95_limit(limits)}")
    if (
        model_server_mean is not None
        and model_server_mean > float(limits["max_model_server_mean_latency_ms_per_request"])
    ):
        errors.append(
            "model_server_mean_latency_ms_per_request "
            f"{model_server_mean} exceeds limit {limits['max_model_server_mean_latency_ms_per_request']}"
        )
    if ram_mb is not None and ram_mb > float(limits["max_ram_mb"]):
        errors.append(f"ram_mb {ram_mb} exceeds limit {limits['max_ram_mb']}")
    if gpu_utilization is not None and gpu_utilization > float(limits["max_gpu_utilization_percent"]):
        errors.append(
            f"gpu_utilization_percent {gpu_utilization} exceeds limit "
            f"{limits['max_gpu_utilization_percent']}"
        )
    if stream_restarts is not None and stream_restarts != 0:
        errors.append(f"stream_restarts must be 0, got {stream_restarts}")
    if false_positive_count is not None and false_positive_count != 0:
        errors.append(f"false_positive_count must be 0, got {false_positive_count}")

    per_class_alerts = _metric(soak, "per_class_alert_count")
    if per_class_alerts is not None and not isinstance(per_class_alerts, dict):
        errors.append("per_class_alert_count must be a mapping when present")
    required_alert_counts = _validate_required_alert_capabilities(per_class_alerts, limits, errors)
    if required_alert_counts:
        summary["required_positive_alert_capabilities"] = required_alert_counts
    suppression_summary = _validate_detector_window_suppression(
        _metric(soak, "detector_window_suppression"),
        limits,
        errors,
    )
    if suppression_summary:
        summary["detector_window_suppression"] = suppression_summary
    guard_summary = _validate_false_positive_guards(
        _metric(soak, "false_positive_guard"),
        limits,
        errors,
    )
    if guard_summary:
        summary["false_positive_guard"] = guard_summary
    return summary


def _recommended_confidence(pack: dict[str, Any], model: str | None) -> float:
    registry_models = pack.get("registry_models") if isinstance(pack.get("registry_models"), dict) else {}
    for entry in registry_models.values():
        if not isinstance(entry, dict):
            continue
        if model and not _model_matches(str(entry.get("file") or ""), model):
            continue
        try:
            return float(entry.get("recommended_confidence"))
        except (TypeError, ValueError):
            continue
    return 0.35


def _expected_input_size(pack: dict[str, Any], model: str | None) -> int:
    registry_models = pack.get("registry_models") if isinstance(pack.get("registry_models"), dict) else {}
    for entry in registry_models.values():
        if not isinstance(entry, dict):
            continue
        if model and not _model_matches(str(entry.get("file") or ""), model):
            continue
        try:
            return int(entry.get("expected_input_size"))
        except (TypeError, ValueError):
            continue
    return 640


def _planned_model_key(pack: dict[str, Any], model: str | None) -> str:
    runtime_handoff = pack.get("runtime_handoff")
    if isinstance(runtime_handoff, dict) and runtime_handoff.get("planned_model_key"):
        return str(runtime_handoff["planned_model_key"])
    registry_models = pack.get("registry_models") if isinstance(pack.get("registry_models"), dict) else {}
    for model_key, entry in registry_models.items():
        if isinstance(entry, dict) and model and _model_matches(str(entry.get("file") or ""), model):
            return str(model_key)
    model_keys = pack.get("model_keys") if isinstance(pack.get("model_keys"), list) else []
    if model_keys:
        return str(model_keys[-1])
    return "model_under_test"


def _normalize_class_name(value: str) -> str:
    return " ".join(str(value).replace("_", " ").replace("-", " ").lower().split())


def _result_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("evidence")
    return evidence if isinstance(evidence, dict) else payload


def _count_from_result(evidence: dict[str, Any], count_key: str, list_key: str) -> float | None:
    raw_count = evidence.get(count_key)
    if raw_count is not None:
        try:
            return float(raw_count)
        except (TypeError, ValueError):
            return None
    raw_list = evidence.get(list_key)
    if isinstance(raw_list, list):
        return float(len(raw_list))
    return None


def _matching_alert_count(evidence: dict[str, Any]) -> float | None:
    return _count_from_result(evidence, "matching_alert_count", "matching_alerts")


def _unexpected_alert_count(evidence: dict[str, Any]) -> float | None:
    return _count_from_result(evidence, "unexpected_alert_count", "unexpected_alerts")


def _max_detections_count(evidence: dict[str, Any]) -> float | None:
    for key in ("max_detections_count", "max_detections", "detections_count", "detections"):
        value = evidence.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    counts = evidence.get("max_detection_class_counts")
    if isinstance(counts, dict):
        values = _numeric_values(counts)
        if values:
            return float(sum(values))
    final_camera = evidence.get("final_camera")
    if isinstance(final_camera, dict):
        counts = final_camera.get("recentDetectionClassCountsMax")
        if isinstance(counts, dict):
            values = _numeric_values(counts)
            if values:
                return float(sum(values))
        value = final_camera.get("detectionsCount")
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _class_counts(evidence: dict[str, Any]) -> dict[str, float]:
    candidates: list[Any] = []
    analytics = evidence.get("analytics_summary")
    if isinstance(analytics, dict):
        candidates.append(analytics.get("class_counts"))
    candidates.append(evidence.get("max_detection_class_counts"))
    final_camera = evidence.get("final_camera")
    if isinstance(final_camera, dict):
        candidates.append(final_camera.get("recentDetectionClassCountsMax"))
    for raw_counts in candidates:
        if not isinstance(raw_counts, dict):
            continue
        counts: dict[str, float] = {}
        for class_name, value in raw_counts.items():
            try:
                counts[str(class_name)] = float(value)
            except (TypeError, ValueError):
                continue
        if counts:
            return counts
    return {}


def _visible_class_total_for_capability(capability: str, evidence: dict[str, Any]) -> float | None:
    for key in ("visible_class_total", "visible_positive_class_total", "visible_ppe_class_total"):
        value = evidence.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    hints = {_normalize_class_name(value) for value in VISIBLE_PPE_CLASS_HINTS.get(capability, set())}
    if not hints:
        return None
    total = 0.0
    for class_name, count in _class_counts(evidence).items():
        normalized = _normalize_class_name(class_name)
        if normalized in hints:
            total += count
    return total


def _schedule_telemetry(evidence: dict[str, Any]) -> dict[str, Any]:
    final_camera = evidence.get("final_camera")
    if isinstance(final_camera, dict):
        telemetry = final_camera.get("scheduleTelemetry")
        if isinstance(telemetry, dict):
            return telemetry
    analytics = evidence.get("analytics_summary")
    if isinstance(analytics, dict):
        schedule = analytics.get("schedule")
        if isinstance(schedule, dict):
            return {
                "scheduleState": {
                    "suppressedCapabilities": schedule.get("suppressed_capabilities") or [],
                    "suppressedCount": schedule.get("suppressed_count"),
                },
                "modelInvocationCounts": schedule.get("model_invocations") or {},
            }
    return {}


def _result_suppressed(capability: str, evidence: dict[str, Any]) -> bool:
    telemetry = _schedule_telemetry(evidence)
    state = telemetry.get("scheduleState")
    if not isinstance(state, dict):
        return False
    suppressed = {str(value) for value in (state.get("suppressedCapabilities") or [])}
    if capability in suppressed:
        return True
    capabilities = state.get("capabilities")
    if isinstance(capabilities, dict):
        entry = capabilities.get(capability)
        if isinstance(entry, dict):
            return entry.get("suppressed") is True
    return False


def _result_model_invocations(evidence: dict[str, Any], fallback_model_key: str) -> dict[str, float]:
    telemetry = _schedule_telemetry(evidence)
    raw = telemetry.get("modelInvocationCounts")
    if not isinstance(raw, dict):
        raw = telemetry.get("model_invocations")
    if not isinstance(raw, dict):
        analytics = evidence.get("analytics_summary")
        if isinstance(analytics, dict):
            schedule = analytics.get("schedule")
            if isinstance(schedule, dict):
                raw = schedule.get("model_invocations")
    if not isinstance(raw, dict):
        return {fallback_model_key: 0}
    invocations: dict[str, float] = {}
    for model_key, value in raw.items():
        try:
            invocations[str(model_key)] = float(value)
        except (TypeError, ValueError):
            invocations[str(model_key)] = 1.0
    return invocations


def _result_identity(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "scenario_id": payload.get("scenario_id") or payload.get("id"),
        "status": payload.get("status"),
        "ok": payload.get("ok"),
    }


def _source_result_failure(path: Path, payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if status in (None, "ready_to_sell", "passed", "pass", "ok"):
        return None
    return f"{path}: status={status}"


def _parse_capability_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected capability=path, got: {value}")
        capability, raw_path = value.split("=", 1)
        capability = capability.strip()
        raw_path = raw_path.strip()
        if not capability or not raw_path:
            raise ValueError(f"expected capability=path, got: {value}")
        paths[capability] = Path(raw_path)
    return paths


def _metric_or_none(metrics: dict[str, Any], key: str) -> Any:
    value = _metric(metrics, key)
    return value if value is not None else None


def _candidate_identity_for_template(
    candidate_report_path: Path | None,
    errors: list[str],
) -> tuple[dict[str, Any], str | None, str | None]:
    if candidate_report_path is None:
        return {"present": False}, None, None
    summary = _candidate_report_identity(candidate_report_path, errors)
    return summary, summary.get("selected_export_sha256"), summary.get("sha256")


def build_raw_benchmark_template(
    *,
    pack_id: str,
    pack_path: Path = DEFAULT_MODEL_PACKS,
    model: str | None = None,
    candidate_report_path: Path | None = None,
    model_artifact_sha256: str | None = None,
    candidate_report_sha256: str | None = None,
) -> dict[str, Any]:
    pack = _load_pack(pack_path, pack_id)
    limits = pack.get("jetson_resource_limits") or {}
    identity_errors: list[str] = []
    candidate_report, derived_artifact_sha, derived_candidate_sha = _candidate_identity_for_template(
        candidate_report_path,
        identity_errors,
    )
    model_name = model or "<model-file>"
    max_cameras = int(limits.get("max_cameras") or 1)
    return {
        "template": True,
        "evidence_kind": "jetson_raw_benchmark",
        "generated_at": utc_now(),
        "pack_id": pack_id,
        "pack_status": pack.get("status"),
        "model": model_name,
        "model_artifact_sha256": model_artifact_sha256 or derived_artifact_sha,
        "candidate_report_sha256": candidate_report_sha256 or derived_candidate_sha,
        "candidate_report": candidate_report,
        "instructions": [
            "Fill this with raw model latency measured on the target Jetson-class device.",
            "Keep cuda=true for a production gate; --allow-cpu is local wiring evidence only.",
            "Use the same candidate_report_sha256 and model_artifact_sha256 as the soak, promotion, and registry reports.",
            f"Include at least {max_cameras} representative camera frames.",
        ],
        "device": "REPLACE_WITH_TARGET_DEVICE_NAME",
        "torch": "REPLACE_WITH_TORCH_VERSION",
        "cuda": True,
        "imgsz": _expected_input_size(pack, model),
        "conf": _recommended_confidence(pack, model),
        "frames": [
            {"name": f"cam{index}-bench.jpg", "shape": [0, 0, 3]}
            for index in range(1, max_cameras + 1)
        ],
        "models": [
            {
                "model": model_name,
                "model_artifact_sha256": model_artifact_sha256 or derived_artifact_sha,
                "candidate_report_sha256": candidate_report_sha256 or derived_candidate_sha,
                "mean_ms": 0,
                "median_ms": 0,
                "p95_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "fps_single_stream_estimate": 0,
                "samples": 0,
                "detections_by_frame_last_run": [
                    {"frame": f"cam{index}-bench.jpg", "count": 0}
                    for index in range(1, max_cameras + 1)
                ],
            }
        ],
        "limits": limits,
        "template_warnings": identity_errors,
    }


def build_soak_report_template(
    *,
    pack_id: str,
    pack_path: Path = DEFAULT_MODEL_PACKS,
    model: str | None = None,
    candidate_report_path: Path | None = None,
    model_artifact_sha256: str | None = None,
    candidate_report_sha256: str | None = None,
) -> dict[str, Any]:
    pack = _load_pack(pack_path, pack_id)
    limits = pack.get("jetson_resource_limits") or {}
    identity_errors: list[str] = []
    candidate_report, derived_artifact_sha, derived_candidate_sha = _candidate_identity_for_template(
        candidate_report_path,
        identity_errors,
    )
    model_name = model or "<model-file>"
    max_cameras = int(limits.get("max_cameras") or 1)
    positive_capabilities = [
        str(value) for value in limits.get("required_positive_alert_capabilities") or []
    ]
    suppression_capabilities = [
        str(value) for value in limits.get("required_detector_window_suppression_capabilities") or []
    ]
    guard_capabilities = [
        str(value) for value in limits.get("required_false_positive_guard_capabilities") or []
    ]
    model_key = _planned_model_key(pack, model)
    return {
        "template": True,
        "evidence_kind": "jetson_three_camera_soak",
        "generated_at": utc_now(),
        "pack_id": pack_id,
        "pack_status": pack.get("status"),
        "model": model_name,
        "model_artifact_sha256": model_artifact_sha256 or derived_artifact_sha,
        "candidate_report_sha256": candidate_report_sha256 or derived_candidate_sha,
        "candidate_report": candidate_report,
        "instructions": [
            "Fill this after a continuous target-device soak using the same candidate artifact as the raw benchmark.",
            "Do not edit counts to pass; copy metrics from runtime telemetry, alerts, screenshots/logs, and detector-window runs.",
            "Positive alert counts must be greater than zero for required capabilities.",
            "False-positive guards must show visible PPE evidence and zero matching alerts.",
            "Detector-window suppression must show zero detections, zero alerts, and zero model invocations outside active windows.",
        ],
        "camera_count": max_cameras,
        "soak_minutes": limits.get("soak_minutes", 0),
        "fps_per_camera": {f"cam{index}": 0 for index in range(1, max_cameras + 1)},
        "mean_latency_ms": 0,
        "p95_latency_ms": 0,
        "model_server_mean_latency_ms_per_request": 0,
        "ram_mb": 0,
        "gpu_utilization_percent": 0,
        "per_class_alert_count": {capability: 0 for capability in positive_capabilities},
        "detector_window_suppression": {
            capability: {
                "suppressed": True,
                "suppressed_capabilities": [capability],
                "max_detections": 0,
                "matching_alerts": 0,
                "unexpected_alerts": 0,
                "model_invocations": {model_key: 0},
            }
            for capability in suppression_capabilities
        },
        "false_positive_guard": {
            capability: {
                "visible_class_total": 0,
                "matching_alerts": 0,
                "unexpected_alerts": 0,
                "false_positive_count": 0,
            }
            for capability in guard_capabilities
        },
        "false_positive_count": 0,
        "stream_restarts": 0,
        "limits": limits,
        "template_warnings": identity_errors,
    }


def build_soak_report_from_results(
    *,
    pack_id: str,
    pack_path: Path = DEFAULT_MODEL_PACKS,
    model: str | None = None,
    candidate_report_path: Path | None = None,
    model_artifact_sha256: str | None = None,
    candidate_report_sha256: str | None = None,
    soak_metrics_path: Path,
    active_result_paths: dict[str, Path],
    guard_result_paths: dict[str, Path],
    suppression_result_paths: dict[str, Path],
) -> dict[str, Any]:
    pack = _load_pack(pack_path, pack_id)
    limits = pack.get("jetson_resource_limits") or {}
    identity_errors: list[str] = []
    candidate_report, derived_artifact_sha, derived_candidate_sha = _candidate_identity_for_template(
        candidate_report_path,
        identity_errors,
    )
    metrics = _load_structured(soak_metrics_path)
    model_name = model or str(metrics.get("model") or "<model-file>")
    max_cameras = int(limits.get("max_cameras") or 1)
    model_key = _planned_model_key(pack, model)
    positive_capabilities = [
        str(value) for value in limits.get("required_positive_alert_capabilities") or []
    ]
    suppression_capabilities = [
        str(value) for value in limits.get("required_detector_window_suppression_capabilities") or []
    ]
    guard_capabilities = [
        str(value) for value in limits.get("required_false_positive_guard_capabilities") or []
    ]
    source_results: dict[str, dict[str, dict[str, Any]]] = {
        "active": {},
        "false_positive_guard": {},
        "detector_window_suppression": {},
    }
    source_result_failures: list[str] = []

    per_class_alert_count: dict[str, float | None] = {}
    for capability in positive_capabilities:
        path = active_result_paths.get(capability)
        if path is None:
            per_class_alert_count[capability] = None
            source_result_failures.append(f"missing active result for {capability}")
            continue
        payload = _load_json(path)
        evidence = _result_evidence(payload)
        source_results["active"][capability] = _result_identity(path, payload)
        failure = _source_result_failure(path, payload)
        if failure:
            source_result_failures.append(failure)
        per_class_alert_count[capability] = _matching_alert_count(evidence)

    false_positive_guard: dict[str, dict[str, Any]] = {}
    for capability in guard_capabilities:
        path = guard_result_paths.get(capability)
        if path is None:
            false_positive_guard[capability] = {
                "visible_class_total": None,
                "matching_alerts": None,
                "unexpected_alerts": None,
                "false_positive_count": None,
            }
            source_result_failures.append(f"missing false-positive guard result for {capability}")
            continue
        payload = _load_json(path)
        evidence = _result_evidence(payload)
        source_results["false_positive_guard"][capability] = _result_identity(path, payload)
        failure = _source_result_failure(path, payload)
        if failure:
            source_result_failures.append(failure)
        matching_alerts = _matching_alert_count(evidence)
        unexpected_alerts = _unexpected_alert_count(evidence)
        false_positive_guard[capability] = {
            "visible_class_total": _visible_class_total_for_capability(capability, evidence),
            "matching_alerts": matching_alerts,
            "unexpected_alerts": unexpected_alerts,
            "false_positive_count": (matching_alerts or 0) + (unexpected_alerts or 0)
            if matching_alerts is not None or unexpected_alerts is not None
            else None,
        }

    detector_window_suppression: dict[str, dict[str, Any]] = {}
    for capability in suppression_capabilities:
        path = suppression_result_paths.get(capability)
        if path is None:
            detector_window_suppression[capability] = {
                "suppressed": False,
                "suppressed_capabilities": [],
                "max_detections": None,
                "matching_alerts": None,
                "unexpected_alerts": None,
                "model_invocations": {},
            }
            source_result_failures.append(f"missing detector-window suppression result for {capability}")
            continue
        payload = _load_json(path)
        evidence = _result_evidence(payload)
        source_results["detector_window_suppression"][capability] = _result_identity(path, payload)
        failure = _source_result_failure(path, payload)
        if failure:
            source_result_failures.append(failure)
        detector_window_suppression[capability] = {
            "suppressed": _result_suppressed(capability, evidence),
            "suppressed_capabilities": [capability] if _result_suppressed(capability, evidence) else [],
            "max_detections": _max_detections_count(evidence),
            "matching_alerts": _matching_alert_count(evidence),
            "unexpected_alerts": _unexpected_alert_count(evidence),
            "model_invocations": _result_model_invocations(evidence, model_key),
        }

    derived_false_positive_count = sum(
        float(entry.get("false_positive_count") or 0)
        for entry in false_positive_guard.values()
        if isinstance(entry, dict)
    )
    false_positive_count = _metric_or_none(metrics, "false_positive_count")
    if false_positive_count is None and false_positive_guard:
        false_positive_count = derived_false_positive_count
    camera_count = _metric_or_none(metrics, "camera_count")
    if camera_count is None:
        camera_count = _metric_or_none(metrics, "active_cameras")
    if camera_count is None:
        camera_count = max_cameras

    return {
        "evidence_kind": "jetson_three_camera_soak",
        "generated_at": utc_now(),
        "pack_id": pack_id,
        "pack_status": pack.get("status"),
        "model": model_name,
        "model_artifact_sha256": model_artifact_sha256 or derived_artifact_sha,
        "candidate_report_sha256": candidate_report_sha256 or derived_candidate_sha,
        "candidate_report": candidate_report,
        "source_metrics": str(soak_metrics_path),
        "source_results": source_results,
        "source_result_failures": source_result_failures,
        "camera_count": camera_count,
        "soak_minutes": _metric_or_none(metrics, "soak_minutes"),
        "fps_per_camera": _metric_or_none(metrics, "fps_per_camera"),
        "mean_latency_ms": _metric_or_none(metrics, "mean_latency_ms"),
        "p95_latency_ms": _metric_or_none(metrics, "p95_latency_ms"),
        "model_server_mean_latency_ms_per_request": _metric_or_none(
            metrics,
            "model_server_mean_latency_ms_per_request",
        ),
        "ram_mb": _metric_or_none(metrics, "ram_mb"),
        "gpu_utilization_percent": _metric_or_none(metrics, "gpu_utilization_percent"),
        "per_class_alert_count": per_class_alert_count,
        "detector_window_suppression": detector_window_suppression,
        "false_positive_guard": false_positive_guard,
        "false_positive_count": false_positive_count,
        "stream_restarts": _metric_or_none(metrics, "stream_restarts"),
        "limits": limits,
        "template_warnings": identity_errors,
    }


def validate_jetson_benchmark(
    *,
    pack_id: str,
    pack_path: Path = DEFAULT_MODEL_PACKS,
    raw_benchmark_path: Path | None = None,
    soak_report_path: Path | None = None,
    candidate_report_path: Path | None = None,
    model: str | None = None,
    model_artifact_sha256: str | None = None,
    candidate_report_sha256: str | None = None,
    allow_cpu: bool = False,
    require_full_gate: bool = False,
) -> dict[str, Any]:
    pack = _load_pack(pack_path, pack_id)
    limits = pack.get("jetson_resource_limits") or {}
    errors: list[str] = []
    warnings: list[str] = []

    if not raw_benchmark_path and not soak_report_path:
        errors.append("provide --raw-benchmark, --soak-report, or both")

    raw_summary = {"present": False}
    soak_summary = {"present": False}
    candidate_report_summary = {"present": False}

    expected_model_artifact_sha256 = model_artifact_sha256
    expected_candidate_report_sha256 = candidate_report_sha256
    if candidate_report_path:
        candidate_report_summary = _candidate_report_identity(
            candidate_report_path,
            errors,
            require_seed_source_recheck=pack_id == FACTORY_PPE_PACK_ID,
        )
        derived_model_artifact_sha256 = candidate_report_summary.get("selected_export_sha256")
        derived_candidate_report_sha256 = candidate_report_summary.get("sha256")
        if (
            model_artifact_sha256
            and derived_model_artifact_sha256
            and model_artifact_sha256 != derived_model_artifact_sha256
        ):
            errors.append(
                "--model-artifact-sha256 does not match candidate report "
                "selected_export.sha256"
            )
        if (
            candidate_report_sha256
            and derived_candidate_report_sha256
            and candidate_report_sha256 != derived_candidate_report_sha256
        ):
            errors.append("--candidate-report-sha256 does not match candidate report file sha256")
        expected_model_artifact_sha256 = model_artifact_sha256 or derived_model_artifact_sha256
        expected_candidate_report_sha256 = candidate_report_sha256 or derived_candidate_report_sha256

    if raw_benchmark_path:
        raw_summary = _validate_raw_benchmark(
            _load_json(raw_benchmark_path),
            limits,
            model=model,
            allow_cpu=allow_cpu,
            errors=errors,
            warnings=warnings,
        )
    if soak_report_path:
        soak_summary = _validate_soak_report(
            _load_json(soak_report_path),
            limits,
            pack_id=pack_id,
            model=model,
            errors=errors,
            warnings=warnings,
        )

    raw_artifact_sha = raw_summary.get("model_artifact_sha256") if isinstance(raw_summary, dict) else None
    soak_artifact_sha = soak_summary.get("model_artifact_sha256") if isinstance(soak_summary, dict) else None
    raw_candidate_report_sha = (
        raw_summary.get("candidate_report_sha256") if isinstance(raw_summary, dict) else None
    )
    soak_candidate_report_sha = (
        soak_summary.get("candidate_report_sha256") if isinstance(soak_summary, dict) else None
    )
    if raw_artifact_sha and soak_artifact_sha and raw_artifact_sha != soak_artifact_sha:
        errors.append("raw benchmark model_artifact_sha256 must match soak report model_artifact_sha256")
    if raw_candidate_report_sha and soak_candidate_report_sha and raw_candidate_report_sha != soak_candidate_report_sha:
        errors.append("raw benchmark candidate_report_sha256 must match soak report candidate_report_sha256")
    if expected_model_artifact_sha256:
        if raw_benchmark_path and not raw_artifact_sha:
            errors.append("raw benchmark missing model_artifact_sha256")
        if soak_report_path and not soak_artifact_sha:
            errors.append("soak report missing model_artifact_sha256")
        if raw_artifact_sha and raw_artifact_sha != expected_model_artifact_sha256:
            errors.append("raw benchmark model_artifact_sha256 does not match expected value")
        if soak_artifact_sha and soak_artifact_sha != expected_model_artifact_sha256:
            errors.append("soak report model_artifact_sha256 does not match expected value")
    if expected_candidate_report_sha256:
        if raw_benchmark_path and not raw_candidate_report_sha:
            errors.append("raw benchmark missing candidate_report_sha256")
        if soak_report_path and not soak_candidate_report_sha:
            errors.append("soak report missing candidate_report_sha256")
        if raw_candidate_report_sha and raw_candidate_report_sha != expected_candidate_report_sha256:
            errors.append("raw benchmark candidate_report_sha256 does not match expected value")
        if soak_candidate_report_sha and soak_candidate_report_sha != expected_candidate_report_sha256:
            errors.append("soak report candidate_report_sha256 does not match expected value")

    if require_full_gate:
        if not raw_benchmark_path:
            errors.append("full Jetson gate requires --raw-benchmark")
        if not soak_report_path:
            errors.append("full Jetson gate requires --soak-report")
        if pack_id == FACTORY_PPE_PACK_ID and not candidate_report_path:
            errors.append("factory_ppe_3cam full Jetson gate requires --candidate-report")

    raw_ready = bool(raw_benchmark_path) and not any(error.startswith("raw benchmark") for error in errors)
    soak_ready = bool(soak_report_path) and not any(error.startswith("soak report") for error in errors)
    production_gate = raw_ready and soak_ready and not errors
    if errors:
        gate_status = "not_ready"
    elif production_gate:
        gate_status = "jetson_gate_passed"
    elif raw_ready and not soak_report_path:
        gate_status = "raw_benchmark_passed_soak_missing"
    elif soak_ready and not raw_benchmark_path:
        gate_status = "soak_report_passed_raw_missing"
    else:
        gate_status = "partial_evidence_only"

    if not production_gate and not require_full_gate:
        warnings.append("production Jetson gate remains blocked until raw benchmark and 3-camera soak both pass")

    return {
        "ok": not errors,
        "generated_at": utc_now(),
        "pack_id": pack_id,
        "pack_status": pack.get("status"),
        "model": model,
        "model_artifact_sha256": expected_model_artifact_sha256 or raw_artifact_sha or soak_artifact_sha,
        "candidate_report_sha256": expected_candidate_report_sha256
        or raw_candidate_report_sha
        or soak_candidate_report_sha,
        "gate_status": gate_status,
        "production_gate": production_gate,
        "errors": errors,
        "warnings": warnings,
        "limits": limits,
        "inputs": {
            "model_pack": str(pack_path),
            "raw_benchmark": str(raw_benchmark_path) if raw_benchmark_path else None,
            "soak_report": str(soak_report_path) if soak_report_path else None,
            "candidate_report": str(candidate_report_path) if candidate_report_path else None,
            "model_artifact_sha256": model_artifact_sha256,
            "candidate_report_sha256": candidate_report_sha256,
        },
        "input_file_sha256s": {
            "candidate_report": candidate_report_summary.get("sha256") if candidate_report_path else None,
            "raw_benchmark": _sha256_file(raw_benchmark_path) if raw_benchmark_path else None,
            "soak_report": _sha256_file(soak_report_path) if soak_report_path else None,
        },
        "candidate_report": candidate_report_summary,
        "raw_benchmark": raw_summary,
        "soak_report": soak_summary,
        "next_required_gates": [] if production_gate else [
            "run_raw_cuda_model_benchmark",
            "run_three_camera_soak_report",
            "rerun_jetson_benchmark_doctor_with_require_full_gate",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Jetson benchmark evidence for a model pack.")
    parser.add_argument("--pack", required=True, help="Model pack id from qa/video_eval/model_packs.yaml")
    parser.add_argument("--model-pack", default=str(DEFAULT_MODEL_PACKS), help="Path to model_packs.yaml")
    parser.add_argument("--model", default="", help="Expected benchmark model filename or path")
    parser.add_argument("--candidate-report", default="", help="JSON from scripts/apron_harness_candidate_doctor.py")
    parser.add_argument("--model-artifact-sha256", default="", help="Expected selected export SHA256 for the benchmarked model artifact")
    parser.add_argument("--candidate-report-sha256", default="", help="Expected candidate report SHA256 for the benchmarked model artifact")
    parser.add_argument("--raw-benchmark", default="", help="JSON from scripts/benchmark_yolo_jetson.py")
    parser.add_argument("--soak-report", default="", help="Three-camera soak report JSON")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU raw benchmark for local wiring only")
    parser.add_argument("--require-full-gate", action="store_true", help="Fail unless raw and soak reports both pass")
    parser.add_argument("--write-raw-template", default="", help="Write a fillable raw benchmark JSON template and exit")
    parser.add_argument("--write-soak-template", default="", help="Write a fillable three-camera soak JSON template and exit")
    parser.add_argument("--build-soak-report", default="", help="Build a three-camera soak report JSON from metrics and scenario result files")
    parser.add_argument("--soak-metrics", default="", help="JSON/YAML with measured target-device soak metrics")
    parser.add_argument("--active-result", action="append", default=[], help="Positive scenario result as capability=path")
    parser.add_argument("--guard-result", action="append", default=[], help="False-positive guard result as capability=path")
    parser.add_argument("--suppression-result", action="append", default=[], help="Detector-window suppression result as capability=path")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_raw_template or args.write_soak_template:
        outputs: dict[str, Any] = {}
        if args.write_raw_template:
            raw_template = build_raw_benchmark_template(
                pack_id=args.pack,
                pack_path=Path(args.model_pack),
                model=args.model or None,
                candidate_report_path=Path(args.candidate_report) if args.candidate_report else None,
                model_artifact_sha256=args.model_artifact_sha256 or None,
                candidate_report_sha256=args.candidate_report_sha256 or None,
            )
            raw_out = Path(args.write_raw_template)
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            raw_out.write_text(json.dumps(raw_template, indent=2) + "\n", encoding="utf-8")
            outputs["raw_template"] = str(raw_out)
        if args.write_soak_template:
            soak_template = build_soak_report_template(
                pack_id=args.pack,
                pack_path=Path(args.model_pack),
                model=args.model or None,
                candidate_report_path=Path(args.candidate_report) if args.candidate_report else None,
                model_artifact_sha256=args.model_artifact_sha256 or None,
                candidate_report_sha256=args.candidate_report_sha256 or None,
            )
            soak_out = Path(args.write_soak_template)
            soak_out.parent.mkdir(parents=True, exist_ok=True)
            soak_out.write_text(json.dumps(soak_template, indent=2) + "\n", encoding="utf-8")
            outputs["soak_template"] = str(soak_out)
        if args.json:
            print(json.dumps(outputs, indent=2))
        else:
            for label, path in outputs.items():
                print(f"wrote {label}: {path}")
        return 0

    if args.build_soak_report:
        if not args.soak_metrics:
            print("ERROR: --build-soak-report requires --soak-metrics")
            return 2
        try:
            active_result_paths = _parse_capability_paths(args.active_result)
            guard_result_paths = _parse_capability_paths(args.guard_result)
            suppression_result_paths = _parse_capability_paths(args.suppression_result)
            soak_report = build_soak_report_from_results(
                pack_id=args.pack,
                pack_path=Path(args.model_pack),
                model=args.model or None,
                candidate_report_path=Path(args.candidate_report) if args.candidate_report else None,
                model_artifact_sha256=args.model_artifact_sha256 or None,
                candidate_report_sha256=args.candidate_report_sha256 or None,
                soak_metrics_path=Path(args.soak_metrics),
                active_result_paths=active_result_paths,
                guard_result_paths=guard_result_paths,
                suppression_result_paths=suppression_result_paths,
            )
        except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            print(f"ERROR: {exc}")
            return 2
        soak_out = Path(args.build_soak_report)
        soak_out.parent.mkdir(parents=True, exist_ok=True)
        soak_out.write_text(json.dumps(soak_report, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps({"soak_report": str(soak_out)}, indent=2))
        else:
            print(f"wrote soak_report: {soak_out}")
        return 0

    report = validate_jetson_benchmark(
        pack_id=args.pack,
        pack_path=Path(args.model_pack),
        raw_benchmark_path=Path(args.raw_benchmark) if args.raw_benchmark else None,
        soak_report_path=Path(args.soak_report) if args.soak_report else None,
        candidate_report_path=Path(args.candidate_report) if args.candidate_report else None,
        model=args.model or None,
        model_artifact_sha256=args.model_artifact_sha256 or None,
        candidate_report_sha256=args.candidate_report_sha256 or None,
        allow_cpu=args.allow_cpu,
        require_full_gate=args.require_full_gate,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        print(f"{status}: {report['pack_id']} {report['gate_status']}")
        if args.out:
            print(f"wrote: {args.out}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
