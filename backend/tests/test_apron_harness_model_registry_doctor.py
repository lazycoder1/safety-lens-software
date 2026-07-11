"""Tests for closed-set apron/harness model registry handoff."""

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json


ROOT = Path(__file__).resolve().parents[2]
DOCTOR_PATH = ROOT / "scripts" / "apron_harness_model_registry_doctor.py"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("apron_harness_model_registry_doctor", DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_report(export_path: Path, export_sha256: str) -> dict:
    return {
        "ok": True,
        "promotion_manifest": {
            "candidate_status": "ready_for_side_by_side_runtime_test",
            "runtime_handoff": {
                "planned_model_key": "ppe_closed_set_candidate",
                "planned_registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                "selected_export": {
                    "path": str(export_path),
                    "exists": True,
                    "accepted_suffix": True,
                    "suffix": export_path.suffix,
                    "sha256": export_sha256,
                },
                "registry_entry": {
                    "model_key": "ppe_closed_set_candidate",
                    "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                    "source_export_sha256": export_sha256,
                },
            },
        },
    }


def _seed_export_import_manifest(
    *,
    updated_manifest_sha256: str = "b" * 64,
    partial_materialization: bool = False,
) -> dict:
    return {
        "path": "/cleared/apron_harness_capture_manifest.yaml.seed_export_import.json",
        "exists": True,
        "checked": True,
        "valid": True,
        "sha256": "5" * 64,
        "seed_source_review_sha256": "6" * 64,
        "seed_import_manifest_sha256": "3" * 64,
        "source_recheck": {
            "path": "qa/video_eval/results/apron_harness_source_recheck_2026_06_24.md",
            "exists": True,
            "sha256": "7" * 64,
            "evidence_boundary": "Fresh source research evidence only; this does not approve training.",
        },
        "updated_manifest_sha256": updated_manifest_sha256,
        "imported_label_count": 3,
        "imported_clip_count": 3,
        "copied_image_count": 3,
        "partial_materialization": partial_materialization,
        "updated_manifest_validation": {
            "checked": True,
            "mode": "production",
            "schema_only": False,
            "ok": True,
            "manifest_sha256": updated_manifest_sha256,
            "errors": [],
            "warnings": [],
        },
        "imports": [
            {
                "source_ref": "roboflow_seed",
                "capability": "harness_required",
                "raw_export_sha256": "4" * 64,
                "imported_label_count": 3,
                "copied_image_count": 3,
                "errors": [],
                "warnings": [],
                "yolo_export_preflight": {
                    "checked": True,
                    "sha256": "4" * 64,
                    "source_classes": ["person", "safety-harness", "lanyard"],
                    "mapped_local_classes": ["person", "safety_harness", "safety_lanyard"],
                    "image_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "label_file_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "orphan_label_count_by_split": {},
                    "label_file_count_by_local_class": {
                        "person": 3,
                        "safety_harness": 3,
                        "safety_lanyard": 3,
                    },
                    "errors": [],
                    "warnings": [],
                },
            }
        ],
    }


def _candidate_report_with_required_seed_import(
    export_path: Path,
    export_sha256: str,
    *,
    seed_export_import_manifest: dict | None = None,
) -> dict:
    report = _candidate_report(export_path, export_sha256)
    report["promotion_manifest"]["training_source_lineage"] = {
        "capture_manifest": {
            "file": {
                "required": True,
                "path": "/cleared/apron_harness_capture_manifest.yaml",
                "exists": True,
                "sha256": "b" * 64,
            },
            "manifest_sha256": "b" * 64,
        },
        "seed_import_manifest": {
            "file": {
                "required": True,
                "path": "/cleared/seed_import.yaml",
                "exists": True,
                "sha256": "3" * 64,
            },
            "gate": {
                "required": True,
                "ok": True,
                "approved_clip_count": 1,
            },
        },
    }
    if seed_export_import_manifest is not None:
        report["promotion_manifest"]["seed_export_import_manifest"] = seed_export_import_manifest
    return report


def _write_promotion_reports(
    apron_path: Path,
    harness_path: Path,
    *,
    candidate_report_path: Path,
    export_sha256: str,
    harness_export_sha256: str | None = None,
) -> None:
    candidate_report_sha256 = _sha256_file(candidate_report_path)
    for capability, path, selected_sha in (
        ("apron_required", apron_path, export_sha256),
        ("harness_required", harness_path, harness_export_sha256 or export_sha256),
    ):
        _write_json(
            path,
            {
                "ok": True,
                "capability": capability,
                "promotion_status": "ready_for_runtime_registration",
                "candidate_report_sha256": candidate_report_sha256,
                "candidate_selected_export": {
                    "path": "/cleared/runs/apron-harness-ppe.onnx",
                    "exists": True,
                    "accepted_suffix": True,
                    "suffix": ".onnx",
                    "sha256": selected_sha,
                },
            },
        )


def test_registry_doctor_dry_run_validates_candidate_without_copy(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report(export_path, _sha256_bytes(export_bytes)))

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
    )

    assert report["ok"] is True
    assert report["registry_status"] == "ready_to_copy"
    assert report["copy_requested"] is False
    assert report["copied"] is False
    assert report["source_export"]["sha256"] == _sha256_bytes(export_bytes)
    assert report["destination"]["exists"] is False
    assert report["model_manager_definition"]["valid"] is True
    assert report["destination"]["registry_metadata"]["exists"] is False
    assert "dry run only" in report["warnings"][0]


def test_registry_doctor_planned_audit_does_not_register_without_candidate(tmp_path: Path):
    doctor = _load_doctor()

    report = doctor.audit_planned_registry(registry_root=tmp_path / "registry")

    assert report["ok"] is True
    assert report["registry_status"] == "planned_no_candidate"
    assert report["candidate_report"] is None
    assert report["candidate_report_sha256"] is None
    assert report["copy_requested"] is False
    assert report["copied"] is False
    assert report["model_manager_definition"]["valid"] is True
    assert report["destination"]["exists"] is False
    assert report["destination"]["matches_expected_sha256"] is False
    assert report["registry_entry"]["model_key"] == "ppe_closed_set_candidate"
    assert "run_apron_harness_candidate_doctor" in report["next_required_gates"]
    assert any("planned-registry audit only" in warning for warning in report["warnings"])


def test_registry_doctor_copies_and_verifies_candidate_artifact(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    expected_sha = _sha256_bytes(export_bytes)
    _write_json(candidate_path, _candidate_report(export_path, expected_sha))
    apron_promotion = tmp_path / "apron_promotion.json"
    harness_promotion = tmp_path / "harness_promotion.json"
    _write_promotion_reports(
        apron_promotion,
        harness_promotion,
        candidate_report_path=candidate_path,
        export_sha256=expected_sha,
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        apron_promotion_report=apron_promotion,
        harness_promotion_report=harness_promotion,
        copy=True,
    )

    destination = tmp_path / "registry" / "models" / "ppe_closed_set_candidate" / "apron-harness-ppe.onnx"
    metadata_path = destination.with_suffix(destination.suffix + ".registry.json")
    assert report["ok"] is True
    assert report["registry_status"] == "registered"
    assert report["copied"] is True
    assert destination.read_bytes() == export_bytes
    assert metadata_path.exists()
    assert report["destination"]["sha256"] == expected_sha
    assert report["destination"]["matches_expected_sha256"] is True
    assert report["destination"]["artifact_matches_expected_sha256"] is True
    assert report["destination"]["registry_metadata"]["valid"] is True
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["kind"] == doctor.REGISTRY_METADATA_KIND
    assert metadata["model_key"] == "ppe_closed_set_candidate"
    assert metadata["artifact_sha256"] == expected_sha
    assert metadata["candidate_report_sha256"] == report["candidate_report_sha256"]
    assert metadata["source_export_sha256"] == expected_sha
    assert metadata["seed_export_import_manifest_required"] is False
    assert metadata["seed_export_import_manifest_sha256"] is None


def test_registry_doctor_rejects_copy_without_promotion_reports(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    expected_sha = _sha256_bytes(export_bytes)
    _write_json(candidate_path, _candidate_report(export_path, expected_sha))

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        apron_promotion_report=tmp_path / "missing_apron_promotion.json",
        harness_promotion_report=tmp_path / "missing_harness_promotion.json",
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert report["copied"] is False
    assert report["promotion_reports"]["apron_required"]["present"] is False
    assert report["promotion_reports"]["harness_required"]["present"] is False
    assert any("apron_required promotion report is required before registry copy" in error for error in report["errors"])
    assert any("harness_required promotion report is required before registry copy" in error for error in report["errors"])


def test_registry_doctor_rejects_copy_when_promotion_export_sha_mismatches(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    expected_sha = _sha256_bytes(export_bytes)
    apron_promotion = tmp_path / "apron_promotion.json"
    harness_promotion = tmp_path / "harness_promotion.json"
    _write_json(candidate_path, _candidate_report(export_path, expected_sha))
    _write_promotion_reports(
        apron_promotion,
        harness_promotion,
        candidate_report_path=candidate_path,
        export_sha256=expected_sha,
        harness_export_sha256="e" * 64,
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        apron_promotion_report=apron_promotion,
        harness_promotion_report=harness_promotion,
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert report["copied"] is False
    assert report["promotion_reports"]["apron_required"]["valid"] is True
    assert report["promotion_reports"]["harness_required"]["valid"] is False
    assert any(
        "harness_required promotion report candidate_selected_export.sha256 must match candidate report" in error
        for error in report["errors"]
    )


def test_registry_doctor_copies_seed_import_lineage_into_metadata(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    expected_sha = _sha256_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    _write_json(
        candidate_path,
        _candidate_report_with_required_seed_import(
            export_path,
            expected_sha,
            seed_export_import_manifest=_seed_export_import_manifest(),
        ),
    )
    apron_promotion = tmp_path / "apron_promotion.json"
    harness_promotion = tmp_path / "harness_promotion.json"
    _write_promotion_reports(
        apron_promotion,
        harness_promotion,
        candidate_report_path=candidate_path,
        export_sha256=expected_sha,
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        apron_promotion_report=apron_promotion,
        harness_promotion_report=harness_promotion,
        copy=True,
    )

    destination = tmp_path / "registry" / "models" / "ppe_closed_set_candidate" / "apron-harness-ppe.onnx"
    metadata = json.loads(destination.with_suffix(destination.suffix + ".registry.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["registry_status"] == "registered"
    assert report["seed_export_import_manifest"]["required"] is True
    assert report["seed_export_import_manifest"]["valid"] is True
    assert metadata["seed_export_import_manifest_required"] is True
    assert metadata["seed_export_import_manifest_sha256"] == "5" * 64
    assert metadata["seed_source_review_sha256"] == "6" * 64
    assert metadata["seed_import_manifest_sha256"] == "3" * 64
    assert metadata["seed_updated_manifest_sha256"] == "b" * 64
    assert metadata["seed_source_recheck"]["sha256"] == "7" * 64
    assert report["seed_export_import_manifest"]["source_recheck"]["path"].endswith(
        "apron_harness_source_recheck_2026_06_24.md"
    )


def test_registry_doctor_rejects_required_seed_import_without_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    _write_json(
        candidate_path,
        _candidate_report_with_required_seed_import(export_path, _sha256_bytes(export_bytes)),
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert report["seed_export_import_manifest"]["required"] is True
    assert report["seed_export_import_manifest"]["present"] is False
    assert any("seed_export_import_manifest is required" in error for error in report["errors"])


def test_registry_doctor_rejects_stale_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    _write_json(
        candidate_path,
        _candidate_report_with_required_seed_import(
            export_path,
            _sha256_bytes(export_bytes),
            seed_export_import_manifest=_seed_export_import_manifest(updated_manifest_sha256="9" * 64),
        ),
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert any("updated_manifest_sha256 must match" in error for error in report["errors"])


def test_registry_doctor_rejects_seed_export_sidecar_without_source_recheck(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    seed_sidecar = _seed_export_import_manifest()
    seed_sidecar.pop("source_recheck")
    _write_json(
        candidate_path,
        _candidate_report_with_required_seed_import(
            export_path,
            _sha256_bytes(export_bytes),
            seed_export_import_manifest=seed_sidecar,
        ),
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert any("seed_export_import_manifest.source_recheck is required" in error for error in report["errors"])


def test_registry_doctor_rejects_partial_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    _write_json(
        candidate_path,
        _candidate_report_with_required_seed_import(
            export_path,
            _sha256_bytes(export_bytes),
            seed_export_import_manifest=_seed_export_import_manifest(partial_materialization=True),
        ),
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert any("partial_materialization must be false" in error for error in report["errors"])


def test_registry_doctor_rejects_seed_export_sidecar_without_yolo_preflight(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    seed_sidecar = _seed_export_import_manifest()
    seed_sidecar["imports"][0].pop("yolo_export_preflight")
    _write_json(
        candidate_path,
        _candidate_report_with_required_seed_import(
            export_path,
            _sha256_bytes(export_bytes),
            seed_export_import_manifest=seed_sidecar,
        ),
    )

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert any(
        "seed_export_import_manifest.imports[0].yolo_export_preflight is required" in error
        for error in report["errors"]
    )


def test_registry_doctor_requires_metadata_for_existing_destination(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake onnx export"
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(export_bytes)
    expected_sha = _sha256_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report(export_path, expected_sha))
    apron_promotion = tmp_path / "apron_promotion.json"
    harness_promotion = tmp_path / "harness_promotion.json"
    _write_promotion_reports(
        apron_promotion,
        harness_promotion,
        candidate_report_path=candidate_path,
        export_sha256=expected_sha,
    )
    destination = tmp_path / "registry" / "models" / "ppe_closed_set_candidate" / "apron-harness-ppe.onnx"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(export_bytes)

    dry_run = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
    )
    copied = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        apron_promotion_report=apron_promotion,
        harness_promotion_report=harness_promotion,
        copy=True,
    )

    assert dry_run["ok"] is True
    assert dry_run["registry_status"] == "ready_to_copy"
    assert dry_run["destination"]["artifact_matches_expected_sha256"] is True
    assert dry_run["destination"]["matches_expected_sha256"] is False
    assert any("registry metadata is missing or stale" in warning for warning in dry_run["warnings"])
    assert copied["ok"] is True
    assert copied["registry_status"] == "registered"
    assert copied["copied"] is False
    assert copied["destination"]["registry_metadata"]["valid"] is True


def test_registry_doctor_rejects_source_sha_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"real export")
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report(export_path, "d" * 64))

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
        copy=True,
    )

    assert report["ok"] is False
    assert report["registry_status"] == "not_ready"
    assert any("selected export sha256 does not match" in error for error in report["errors"])


def test_registry_doctor_rejects_engine_export_for_onnx_registry_path(tmp_path: Path):
    doctor = _load_doctor()
    export_bytes = b"fake tensorrt export"
    export_path = tmp_path / "best.engine"
    export_path.write_bytes(export_bytes)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report(export_path, _sha256_bytes(export_bytes)))

    report = doctor.audit_candidate_registry(
        candidate_path,
        registry_root=tmp_path / "registry",
    )

    assert report["ok"] is False
    assert any("selected export suffix must match" in error for error in report["errors"])
