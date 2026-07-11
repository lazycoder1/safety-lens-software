"""Tests for apron/harness public seed-source review gate."""

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import importlib.util

import yaml


ROOT = Path(__file__).resolve().parents[2]
DOCTOR_PATH = ROOT / "scripts" / "apron_harness_seed_source_doctor.py"
MODEL_PACKS_PATH = ROOT / "qa" / "video_eval" / "model_packs.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_yolo_export_zip(
    path: Path,
    *,
    names: list[str],
    label_classes: list[str],
) -> Path:
    class_ids = {name: index for index, name in enumerate(names)}
    label_lines = "\n".join(
        f"{class_ids[class_name]} 0.5 0.5 0.2 0.2"
        for class_name in label_classes
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data.yaml",
            yaml.safe_dump(
                {
                    "train": "train/images",
                    "val": "valid/images",
                    "test": "test/images",
                    "names": names,
                },
                sort_keys=False,
            ),
        )
        for split in ("train", "valid", "test"):
            archive.writestr(f"{split}/images/{split}_000001.jpg", b"fake-image")
            archive.writestr(f"{split}/labels/{split}_000001.txt", label_lines + "\n")
    return path


def _write_orphan_label_yolo_export_zip(
    path: Path,
    *,
    names: list[str],
    label_classes: list[str],
) -> Path:
    class_ids = {name: index for index, name in enumerate(names)}
    label_lines = "\n".join(
        f"{class_ids[class_name]} 0.5 0.5 0.2 0.2"
        for class_name in label_classes
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data.yaml",
            yaml.safe_dump(
                {
                    "train": "train/images",
                    "val": "valid/images",
                    "test": "test/images",
                    "names": names,
                },
                sort_keys=False,
            ),
        )
        for split in ("train", "valid", "test"):
            archive.writestr(f"{split}/images/{split}_image_only.jpg", b"fake-image")
            archive.writestr(f"{split}/labels/{split}_orphan_label.txt", label_lines + "\n")
    return path


def _review_evidence_payload(
    *,
    review_fields: set[str],
    source_ref: str,
    capability: str,
    source_url: str = "https://example.com/source",
    license_note: str = "CC BY 4.0",
    reviewed_by: str = "qa_reviewer",
    reviewed_at: str = "2026-06-22T00:00:00+00:00",
) -> dict:
    return {
        "kind": "apron_harness_seed_source_review_evidence",
        "version": 1,
        "source_ref": source_ref,
        "capability": capability,
        "source_url": source_url,
        "license_note": license_note,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "review_items": {
            field: {
                "approved": True,
                "evidence_ref": f"reviewed/{source_ref}/{capability}/{field}.md",
            }
            for field in sorted(review_fields)
        },
    }


def _write_review_evidence(
    path: Path,
    *,
    review_fields: set[str],
    source_ref: str,
    capability: str,
    source_url: str = "https://example.com/source",
    license_note: str = "CC BY 4.0",
    reviewed_by: str = "qa_reviewer",
    reviewed_at: str = "2026-06-22T00:00:00+00:00",
) -> tuple[str, str]:
    path.write_text(
        yaml.safe_dump(
            _review_evidence_payload(
                review_fields=review_fields,
                source_ref=source_ref,
                capability=capability,
                source_url=source_url,
                license_note=license_note,
                reviewed_by=reviewed_by,
                reviewed_at=reviewed_at,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return str(path), _sha256_file(path)


def _attach_candidate_review_evidence(candidate: dict, tmp_path: Path, review_fields: set[str]) -> None:
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    source = (model_packs.get("shared_sources") or {}).get(candidate["source_ref"]) or {}
    evidence_path, evidence_sha256 = _write_review_evidence(
        tmp_path / f"{candidate['source_ref']}_{candidate['capability']}_review.yaml",
        review_fields=review_fields,
        source_ref=candidate["source_ref"],
        capability=candidate["capability"],
        source_url=source.get("url") or "https://example.com/source",
        license_note=source.get("license_note") or "CC BY 4.0",
        reviewed_by=candidate.get("reviewed_by") or "qa_reviewer",
        reviewed_at=candidate.get("reviewed_at") or "2026-06-22T00:00:00+00:00",
    )
    candidate["review_evidence_path"] = evidence_path
    candidate["review_evidence_sha256"] = evidence_sha256


def _attach_row_review_evidence(row: dict, tmp_path: Path, review_fields: set[str]) -> None:
    evidence_path, evidence_sha256 = _write_review_evidence(
        tmp_path / f"{row['source_ref']}_{row['capability']}_review.yaml",
        review_fields=review_fields,
        source_ref=row["source_ref"],
        capability=row["capability"],
        source_url=row.get("source_url") or "https://example.com/source",
        license_note=row.get("license_note") or "CC BY 4.0",
        reviewed_by=row.get("reviewed_by") or "qa_reviewer",
        reviewed_at=row.get("reviewed_at") or "2026-06-22T00:00:00+00:00",
    )
    row["review_evidence_path"] = evidence_path
    row["review_evidence_sha256"] = evidence_sha256


def _load_doctor():
    spec = importlib.util.spec_from_file_location("apron_harness_seed_source_doctor", DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _approved_harness_seed_report(doctor, tmp_path: Path) -> tuple[dict, dict]:
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")
    report = doctor.audit_seed_sources(path)
    return report, report["candidates"][0]


def _add_generated_review_artifacts(
    doctor,
    report: dict,
    tmp_path: Path,
    *,
    source_ref: str,
    capability: str,
) -> dict:
    evidence_dir = tmp_path / "review_evidence_templates"
    packet_dir = tmp_path / "review_packets"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    artifacts: dict[str, str] = {}
    for item in report["review_packets"]["files"]:
        if item["source_ref"] == source_ref and item["capability"] == capability:
            artifacts["review_packet_path"] = item["path"]
            artifacts["review_packet_sha256"] = item["sha256"]
            break
    for item in report["review_evidence_templates"]["files"]:
        if item["source_ref"] == source_ref and item["capability"] == capability:
            artifacts["review_evidence_template_path"] = item["path"]
            artifacts["review_evidence_template_sha256"] = item["sha256"]
            break
    return artifacts


def _valid_harness_seed_import(
    source: dict,
    *,
    source_review_sha256: str,
    export_zip: Path,
    export_sha256: str,
    review_artifacts: dict | None = None,
) -> dict:
    item = {
        "source_ref": source["source_ref"],
        "capability": source["capability"],
        "include_in_training": True,
        "review_status": "approved_for_training",
        "reviewed_by": "qa_reviewer",
        "reviewed_at": "2026-06-22T00:00:00+00:00",
        "manifest_import_path": source["manifest_import_path"],
        "raw_export_ref": "s3://cleared-seed-exports/harness_seed_yolo_export.zip",
        "raw_export_sha256": export_sha256,
        "raw_export_local_path": str(export_zip),
        "export_format": "yolo",
        "completed_review": {
            "class_mapping": True,
            "dataset_card_provenance": True,
            "export_terms": True,
            "hard_negative_coverage": True,
            "license_terms": True,
            "manifest_import_plan": True,
            "person_box_coverage": True,
            "privacy_and_identity_risk": True,
            "train_val_test_split": True,
        },
        "class_mapping": {
            "top": "person",
            "safety-harness": "safety_harness",
            "lanyard": "safety_lanyard",
        },
        "person_box_policy": "Source top boxes were reviewed and mapped to person boxes.",
        "hard_negative_policy": "Hard negatives are tracked in the capture manifest for non-harness straps.",
        "split_plan": {
            "train": "source-defined train split reconciled to manifest",
            "val": "source-defined val split reconciled to manifest",
            "test": "held-out test split reconciled to manifest",
        },
        "expected_labeled_images_per_class": {
            "person": 3,
            "apron": 0,
            "safety_harness": 3,
            "safety_lanyard": 3,
        },
    }
    if review_artifacts is not None:
        item["review_artifacts"] = review_artifacts
    return {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": source_review_sha256,
        "imports": [item],
    }


def test_seed_source_doctor_blocks_current_unreviewed_public_sources(tmp_path: Path):
    doctor = _load_doctor()
    out = tmp_path / "seed_source_review.md"

    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    out.write_text(doctor.render_work_order(report), encoding="utf-8")
    evidence_dir = tmp_path / "review_evidence_templates"
    written_templates = doctor.write_review_evidence_templates(report, evidence_dir)
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": written_templates,
    }
    packet_dir = tmp_path / "review_packets"
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    import_template = doctor.build_import_manifest_template(
        report,
        seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    checklist_rows = list(csv.DictReader(doctor.build_review_checklist_csv(report).splitlines()))

    assert report["ok"] is True
    assert report["gate_passed"] is False
    assert report["status"] == "blocked_pending_source_review"
    assert report["inputs"]["source_research_max_age_days"] == 14
    assert report["candidate_count"] >= 5
    assert report["training_usable_count"] == 0
    assert set(report["capability_coverage"]) >= {"apron_required", "harness_required"}
    assert set(report["missing_training_capabilities"]) == {"apron_required", "harness_required"}
    assert any("approval_status=unreviewed" in blocker for blocker in report["blockers"])
    high_priority_candidates = [
        candidate
        for candidate in report["candidates"]
        if candidate["review_priority"] <= doctor.SOURCE_FACT_REVIEW_PRIORITY_MAX
    ]
    assert doctor.SOURCE_FACT_REVIEW_PRIORITY_MAX == 50
    assert {candidate["source_ref"] for candidate in high_priority_candidates} >= {
        "roboflow_work_at_height_safety",
        "roboflow_harness_detection",
        "roboflow_work_at_height_safety_d2",
        "roboflow_scaffold_harness",
        "roboflow_safety_harness_public_domain",
        "roboflow_kit_att_det_apron_gloves",
        "roboflow_workspace_otd88_fjwepfj1",
        "roboflow_workspace_otd88_eqjo",
        "roboflow_safety_food_system",
        "roboflow_kitchen_hygiene",
        "roboflow_agts1_newapron",
    }
    assert all(candidate["source_facts"] for candidate in high_priority_candidates)
    assert not any(
        "high-priority source missing source_facts" in warning
        for candidate in high_priority_candidates
        for warning in candidate["warnings"]
    )
    review_queue = report["review_queue_summary"]
    assert review_queue["candidate_count"] == report["candidate_count"]
    assert review_queue["reviewable_count"] == report["candidate_count"] - 1
    assert review_queue["training_usable_count"] == 0
    assert review_queue["unreviewed_count"] == report["candidate_count"] - 1
    assert len(report["review_queue"]) == report["candidate_count"]
    assert report["review_queue"][0]["source_ref"] == "roboflow_work_at_height_safety"
    assert report["review_queue"][0]["capability"] == "harness_required"
    assert report["review_queue"][0]["review_priority"] == 10
    assert report["review_queue"][0]["review_artifacts"]["review_packet_path"] == ""
    assert report["review_queue"][0]["next_action"] == (
        "fill_review_evidence_and_checklist_then_validate_seed_import_manifest"
    )
    assert [
        item["source_ref"]
        for item in review_queue["capabilities"]["harness_required"]["top_review_candidates"]
    ] == [
        "roboflow_work_at_height_safety",
        "roboflow_harness_s4xxh",
        "roboflow_work_at_height_safety_d2",
    ]
    assert [
        item["source_ref"]
        for item in review_queue["capabilities"]["apron_required"]["top_review_candidates"]
    ] == [
        "roboflow_workspace_otd88_fjwepfj1",
        "roboflow_safety_food_system",
        "roboflow_workspace_otd88_eqjo",
    ]
    assert review_queue["capabilities"]["harness_required"]["top_review_candidates"][0]["source_counts"][
        "public_page_image_count"
    ] == 12805
    assert "harness" in review_queue["capabilities"]["harness_required"]["top_review_candidates"][0]["classes"]
    assert review_queue["capabilities"]["harness_required"]["top_review_candidates"][0]["suggested_mapping"][
        "local_class_to_source_labels"
    ]["safety_harness"] == ["harness"]
    assert "01.apron" in review_queue["capabilities"]["apron_required"]["top_review_candidates"][0]["classes"]
    assert review_queue["capabilities"]["apron_required"]["top_review_candidates"][0]["suggested_mapping"][
        "local_class_to_source_labels"
    ]["apron"] == ["01.apron"]
    assert review_queue["capabilities"]["apron_required"]["top_review_candidates"][0]["suggested_mapping"][
        "local_class_to_source_labels"
    ]["person"] == ["00.person"]
    research = report["source_research_readiness"]
    assert research["candidate_count"] == report["candidate_count"]
    assert research["training_usable_count"] == 0
    assert research["evidence_ready_count"] == 24
    assert research["source_research_closed_count"] == 1
    assert research["needs_agent_research_count"] == 0
    assert research["evidence_ready_unapproved_count"] == research["evidence_ready_count"]
    apron_positive_supplement = next(
        candidate
        for candidate in report["candidates"]
        if candidate["source_ref"] == "roboflow_new_workspace_apron_detection_jpnm2"
    )
    assert apron_positive_supplement["capability"] == "apron_required"
    assert apron_positive_supplement["review_priority"] == 72
    assert apron_positive_supplement["source_research_status"]["evidence_ready_for_human_review"] is True
    assert apron_positive_supplement["approval_status"] == "unreviewed"
    assert apron_positive_supplement["training_usable"] is False
    assert "Wearing-Apron" in apron_positive_supplement["source_facts"]["classes"]
    coverage_plan = doctor.build_source_coverage_plan(
        report,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    assert coverage_plan["kind"] == doctor.SOURCE_COVERAGE_PLAN_KIND
    assert coverage_plan["approval_guardrail"]["agent_can_approve_training"] is False
    assert coverage_plan["training_usable_count"] == 0
    assert coverage_plan["coverage_gap_count"] == 0
    apron_coverage = coverage_plan["capabilities"]["apron_required"]
    assert apron_coverage["missing_local_classes_across_reviewable_sources"] == []
    assert apron_coverage["person_box_reconciliation"]["status"] == (
        "candidate_person_mapping_present_pending_review"
    )
    assert apron_coverage["person_box_reconciliation"]["candidate_person_source_refs"] == [
        "roboflow_workspace_otd88_eqjo",
        "roboflow_workspace_otd88_fjwepfj1",
    ]
    assert "roboflow_workspace_otd88_fjwepfj1" in apron_coverage["mapped_class_sources"]["person"]
    assert "roboflow_workspace_otd88_fjwepfj1" in apron_coverage["mapped_class_sources"]["apron"]
    apron_fjwepfj1_coverage = next(
        item
        for item in apron_coverage["candidate_coverage"]
        if item["source_ref"] == "roboflow_workspace_otd88_fjwepfj1"
    )
    assert apron_fjwepfj1_coverage["source_counts"]["image_count"] == 9694
    assert "00.person" in apron_fjwepfj1_coverage["source_classes"]
    assert "01.apron" in apron_fjwepfj1_coverage["source_classes"]
    assert "02.no_apron" in apron_fjwepfj1_coverage["source_classes"]
    assert apron_fjwepfj1_coverage["mapped_local_classes"] == ["apron", "person"]
    assert apron_fjwepfj1_coverage["missing_local_classes"] == []
    assert apron_fjwepfj1_coverage["person_box_status"] == "candidate_person_mapping_present"
    assert "roboflow_safety_food_system" in apron_coverage["person_box_reconciliation"][
        "candidate_without_person_source_refs"
    ]
    assert "roboflow_safety_food_system" in apron_coverage["mapped_class_sources"]["apron"]
    apron_safety_food_coverage = next(
        item
        for item in apron_coverage["candidate_coverage"]
        if item["source_ref"] == "roboflow_safety_food_system"
    )
    assert apron_safety_food_coverage["source_counts"]["image_count"] == 9897
    assert "no_apron" in apron_safety_food_coverage["source_classes"]
    assert apron_safety_food_coverage["person_box_status"] == (
        "missing_person_mapping_requires_reviewed_person_box_reconciliation"
    )
    apron_jpnm2_coverage = next(
        item
        for item in apron_coverage["candidate_coverage"]
        if item["source_ref"] == "roboflow_new_workspace_apron_detection_jpnm2"
    )
    assert apron_jpnm2_coverage["mapped_local_classes"] == ["apron"]
    assert apron_jpnm2_coverage["missing_local_classes"] == ["person"]
    assert "Wearing-Apron" in apron_jpnm2_coverage["source_classes"]
    harness_coverage = coverage_plan["capabilities"]["harness_required"]
    assert harness_coverage["missing_local_classes_across_reviewable_sources"] == []
    assert harness_coverage["person_box_reconciliation"]["status"] == (
        "candidate_person_mapping_present_pending_review"
    )
    assert "roboflow_harness_s4xxh" in harness_coverage["person_box_reconciliation"][
        "candidate_person_source_refs"
    ]
    assert "roboflow_harness_s4xxh" in harness_coverage["complete_single_source_refs"]
    harness_s4xxh_coverage = next(
        item
        for item in harness_coverage["candidate_coverage"]
        if item["source_ref"] == "roboflow_harness_s4xxh"
    )
    assert harness_s4xxh_coverage["source_counts"]["image_count"] == 9802
    assert "lifeline" in harness_s4xxh_coverage["source_classes"]
    assert harness_s4xxh_coverage["person_box_status"] == "candidate_person_mapping_present"
    assert harness_coverage["priority_coverage_plan"]["missing_classes_after_priority_plan"] == []
    assert harness_coverage["priority_coverage_plan"]["status"] == "candidate_coverage_complete_pending_review"
    assert [
        item["source_ref"]
        for item in research["capabilities"]["harness_required"]["evidence_ready_sources"]
    ] == [
        "roboflow_work_at_height_safety",
        "roboflow_harness_s4xxh",
        "roboflow_work_at_height_safety_d2",
        "roboflow_harness_detection",
        "roboflow_safety_harness_iess4",
    ]
    assert [
        item["source_ref"]
        for item in research["capabilities"]["apron_required"]["evidence_ready_sources"]
    ] == [
        "roboflow_workspace_otd88_fjwepfj1",
        "roboflow_safety_food_system",
        "roboflow_workspace_otd88_eqjo",
        "roboflow_kitchen_hygiene",
        "roboflow_agts1_newapron",
    ]
    assert [item["source_ref"] for item in research["source_research_closed_sources"]] == [
        "roboflow_harness_detection_v1"
    ]
    closed_source_status = research["source_research_closed_sources"][0]["source_research_status"]
    assert closed_source_status["status"] == "source_unavailable_after_agent_research"
    assert closed_source_status["agent_research_closed"] is True
    assert closed_source_status["source_unavailable"] is True
    assert research["next_agent_research_sources"] == []
    assert all(
        item["source_research_status"]["evidence_ready_for_human_review"] is True
        for item in research["evidence_ready_sources"]
    )
    harness_detection_v1 = next(
        candidate
        for candidate in report["candidates"]
        if candidate["source_ref"] == "roboflow_harness_detection_v1"
    )
    assert harness_detection_v1["source_research_disposition"] == "unavailable_after_agent_search"
    assert harness_detection_v1["approval_status"] == "rejected"
    assert harness_detection_v1["blocker"] == (
        "source_unavailable_after_agent_search_keep_blocked_or_replace_with_verifiable_source"
    )
    harness_detection_v1_status = doctor._source_research_status(harness_detection_v1)
    assert harness_detection_v1_status["evidence_ready_for_human_review"] is False
    assert harness_detection_v1_status["agent_research_closed"] is True
    assert any(
        "could not verify the public page" in item["finding"]
        or "no discoverable public source page" in item["finding"]
        for item in harness_detection_v1["source_research_evidence"]
    )
    work_at_height = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_work_at_height_safety"
    )
    work_at_height_mapping = work_at_height["suggested_mapping"]
    assert work_at_height_mapping["non_approving"] is True
    assert work_at_height_mapping["local_class_to_source_labels"]["person"] == ["person"]
    assert work_at_height_mapping["local_class_to_source_labels"]["safety_harness"] == ["harness"]
    assert "ladder" in work_at_height_mapping["hard_negative_source_labels"]
    assert any("safety_lanyard" in warning for warning in work_at_height_mapping["warnings"])
    assert work_at_height["source_facts"]["dataset_version"]["id"] == "work-at-height-safety/3"
    assert work_at_height["source_facts"]["dataset_version"]["split"]["train"] == 26892
    assert any("incorrect licenses" in item["finding"] for item in work_at_height["source_research_evidence"])
    assert any(
        "Creative Commons CC BY 4.0 allows commercial sharing/adaptation" in item["finding"]
        for item in work_at_height["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in work_at_height["source_research_evidence"]
    )
    assert any(
        "dataset license is shown in the Cite This Project section" in item["finding"]
        for item in work_at_height["source_research_evidence"]
    )
    harness_s4xxh = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_harness_s4xxh"
    )
    harness_s4xxh_mapping = harness_s4xxh["suggested_mapping"]
    assert harness_s4xxh_mapping["non_approving"] is True
    assert harness_s4xxh_mapping["local_class_to_source_labels"]["person"] == ["Person"]
    assert harness_s4xxh_mapping["local_class_to_source_labels"]["safety_harness"] == ["harness"]
    assert harness_s4xxh_mapping["local_class_to_source_labels"]["safety_lanyard"] == ["lifeline"]
    assert "no harness" in harness_s4xxh_mapping["hard_negative_source_labels"]
    assert "table" in harness_s4xxh["source_facts"]["classes"]
    assert "table" in harness_s4xxh_mapping["ignored_or_unmapped_source_labels"]
    assert any(
        "9,802 images" in item["finding"]
        for item in harness_s4xxh["source_research_evidence"]
    )
    assert any(
        "Cite This Project section lists CC BY 4.0" in item["finding"]
        for item in harness_s4xxh["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in harness_s4xxh["source_research_evidence"]
    )
    work_at_height_d2 = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_work_at_height_safety_d2"
    )
    work_at_height_d2_mapping = work_at_height_d2["suggested_mapping"]
    assert work_at_height_d2_mapping["non_approving"] is True
    assert work_at_height_d2_mapping["local_class_to_source_labels"]["person"] == ["person"]
    assert work_at_height_d2_mapping["local_class_to_source_labels"]["safety_harness"] == ["harness"]
    assert any("safety_lanyard" in warning for warning in work_at_height_d2_mapping["warnings"])
    assert work_at_height_d2["source_facts"]["views"] == 966
    assert work_at_height_d2["source_facts"]["downloads"] == 46
    assert any(
        "2,087 images" in item["finding"]
        for item in work_at_height_d2["source_research_evidence"]
    )
    assert any(
        "preview images show harness and person sample labels" in item["finding"]
        for item in work_at_height_d2["source_research_evidence"]
    )
    assert any(
        "hosted API is not a Jetson Nano local model path" in item["implication"]
        for item in work_at_height_d2["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in work_at_height_d2["source_research_evidence"]
    )
    safety_food = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_safety_food_system"
    )
    assert safety_food["source_facts"]["dataset_version"]["id"] == "safety-food/3"
    assert safety_food["source_facts"]["dataset_version"]["split"]["test"] == 1079
    assert any("apron/no_apron" in item["finding"] for item in safety_food["source_research_evidence"])
    assert any(
        "mAP@50 90.4%" in item["finding"]
        for item in safety_food["source_research_evidence"]
    )
    assert any(
        "Roboflow 3.0 Object Detection (Fast)" in item["finding"]
        for item in safety_food["source_research_evidence"]
    )
    assert any(
        "pest/domain classes may be noisy" in item["implication"]
        for item in safety_food["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in safety_food["source_research_evidence"]
    )
    assert any(
        "reviewer must archive the exact citation block" in item["implication"]
        for item in safety_food["source_research_evidence"]
    )
    fjwepfj1 = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_workspace_otd88_fjwepfj1"
    )
    fjwepfj1_mapping = fjwepfj1["suggested_mapping"]
    assert fjwepfj1["review_priority"] == 14
    assert fjwepfj1["source_facts"]["image_count"] == 9694
    assert fjwepfj1_mapping["non_approving"] is True
    assert fjwepfj1_mapping["local_class_to_source_labels"]["person"] == ["00.person"]
    assert fjwepfj1_mapping["local_class_to_source_labels"]["apron"] == ["01.apron"]
    assert "02.no_apron" in fjwepfj1_mapping["hard_negative_source_labels"]
    assert any(
        "9,694 images" in item["finding"]
        for item in fjwepfj1["source_research_evidence"]
    )
    assert any(
        "Hosted workflow evidence does not replace" in item["implication"]
        for item in fjwepfj1["source_research_evidence"]
    )
    eqjo = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_workspace_otd88_eqjo"
    )
    eqjo_mapping = eqjo["suggested_mapping"]
    assert eqjo["review_priority"] == 16
    assert eqjo["source_facts"]["image_count"] == 4552
    assert eqjo_mapping["non_approving"] is True
    assert eqjo_mapping["local_class_to_source_labels"]["person"] == ["00.person"]
    assert eqjo_mapping["local_class_to_source_labels"]["apron"] == ["01.apron"]
    assert "02.no_apron" in eqjo_mapping["hard_negative_source_labels"]
    assert any(
        "4,552 images" in item["finding"]
        for item in eqjo["source_research_evidence"]
    )
    kitchen_hygiene = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_kitchen_hygiene"
    )
    kitchen_hygiene_mapping = kitchen_hygiene["suggested_mapping"]
    assert kitchen_hygiene_mapping["non_approving"] is True
    assert kitchen_hygiene_mapping["local_class_to_source_labels"]["apron"] == ["apron"]
    assert "no_apron" in kitchen_hygiene_mapping["hard_negative_source_labels"]
    assert kitchen_hygiene["source_facts"]["dataset_version"]["id"] == "kitchenhygiene/2"
    assert kitchen_hygiene["source_facts"]["dataset_version"]["split"]["test"] == 818
    assert any(
        "9,400 visible images" in item["finding"]
        for item in kitchen_hygiene["source_research_evidence"]
    )
    assert any(
        "mAP@50 90.9%" in item["finding"]
        for item in kitchen_hygiene["source_research_evidence"]
    )
    assert any(
        "Roboflow 3.0 Object Detection (Fast)" in item["finding"]
        for item in kitchen_hygiene["source_research_evidence"]
    )
    assert any(
        "food/kitchen-domain apron seed data" in item["implication"]
        for item in kitchen_hygiene["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in kitchen_hygiene["source_research_evidence"]
    )
    agts1_newapron = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_agts1_newapron"
    )
    agts1_newapron_mapping = agts1_newapron["suggested_mapping"]
    assert agts1_newapron_mapping["non_approving"] is True
    assert agts1_newapron_mapping["local_class_to_source_labels"]["apron"] == ["apron"]
    assert "no_apron" in agts1_newapron_mapping["hard_negative_source_labels"]
    assert agts1_newapron["source_facts"]["views"] == 747
    assert agts1_newapron["source_facts"]["downloads"] == 26
    assert any(
        "1,468 images" in item["finding"]
        for item in agts1_newapron["source_research_evidence"]
    )
    assert any(
        "no project description has been published" in item["finding"]
        for item in agts1_newapron["source_research_evidence"]
    )
    assert any(
        "hosted API is not a Jetson Nano local model path" in item["implication"]
        for item in agts1_newapron["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in agts1_newapron["source_research_evidence"]
    )
    kit_att = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_kit_att_det_apron_gloves"
    )
    kit_att_mapping = kit_att["suggested_mapping"]
    assert kit_att_mapping["non_approving"] is True
    assert kit_att_mapping["local_class_to_source_labels"]["apron"] == ["apron"]
    assert "no_apron" in kit_att_mapping["hard_negative_source_labels"]
    assert "gloves" in kit_att_mapping["hard_negative_source_labels"]
    assert "no_headwear" in kit_att_mapping["hard_negative_source_labels"]
    assert kit_att["source_facts"]["views"] == 24
    assert any(
        "1,351 images" in item["finding"]
        for item in kit_att["source_research_evidence"]
    )
    assert any(
        "zero dataset versions" in item["finding"]
        for item in kit_att["source_research_evidence"]
    )
    assert any(
        "hosted API is not a Jetson Nano local model path" in item["implication"]
        for item in kit_att["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in kit_att["source_research_evidence"]
    )
    safety_harness_iess4 = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_safety_harness_iess4"
    )
    assert safety_harness_iess4["url"] == "https://universe.roboflow.com/stage0-wjefs/safety-harness-iess4-zolyk"
    assert safety_harness_iess4["source_facts"]["source_author"] == "stage0"
    assert safety_harness_iess4["source_facts"]["image_count"] == 938
    assert safety_harness_iess4["source_facts"]["classes"] == ["human", "safety-harness"]
    safety_harness_iess4_mapping = safety_harness_iess4["suggested_mapping"]
    assert safety_harness_iess4_mapping["non_approving"] is True
    assert safety_harness_iess4_mapping["local_class_to_source_labels"]["person"] == ["human"]
    assert safety_harness_iess4_mapping["local_class_to_source_labels"]["safety_harness"] == ["safety-harness"]
    assert any("safety_lanyard" in warning for warning in safety_harness_iess4_mapping["warnings"])
    assert any(
        "938 images" in item["finding"]
        for item in safety_harness_iess4["source_research_evidence"]
    )
    assert any(
        "human plus safety-harness sample labels" in item["finding"]
        for item in safety_harness_iess4["source_research_evidence"]
    )
    assert any(
        "Public Plan service use is internal non-commercial" in item["finding"]
        for item in safety_harness_iess4["source_research_evidence"]
    )
    kit_att = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_kit_att_det_apron_gloves"
    )
    assert kit_att["source_facts"]["source_author"] == "Vaanuvaa"
    assert kit_att["source_facts"]["image_count"] == 1351
    assert kit_att["source_facts"]["classes"] == ["apron", "gloves", "no_apron", "no_headwear"]
    ppe_food = next(
        candidate
        for candidate in report["candidates"]
        if candidate["source_ref"] == "roboflow_ppe_food_manufacturing"
    )
    ppe_food_mapping = ppe_food["suggested_mapping"]
    assert ppe_food["source_facts"]["source_author"] == "Stock Hive"
    assert ppe_food["source_facts"]["model_type"] == "YOLOv11 Object Detection (Fast)"
    assert ppe_food["source_facts"]["page_metrics"]["map50"] == "72.2%"
    assert ppe_food_mapping["non_approving"] is True
    assert ppe_food_mapping["local_class_to_source_labels"]["apron"] == ["Apron"]
    assert "Mask" in ppe_food_mapping["hard_negative_source_labels"]
    assert any(
        "Food-manufacturing apron source is domain-relevant" in item["implication"]
        for item in ppe_food["source_research_evidence"]
    )
    assert any(
        "hosted API/checkpoint is not the target Jetson Nano local closed-set path" in item["implication"]
        for item in ppe_food["source_research_evidence"]
    )
    apron_detection = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_apron_detection"
    )
    apron_detection_mapping = apron_detection["suggested_mapping"]
    assert apron_detection["source_facts"]["source_author"] == "knowledgeflex"
    assert apron_detection["source_facts"]["image_count"] == 576
    assert apron_detection_mapping["local_class_to_source_labels"]["apron"] == ["Wearing-Apron"]
    assert any("missing-apron negatives" in item["implication"] for item in apron_detection["source_research_evidence"])
    agts1_apron = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_agts1_apron"
    )
    agts1_apron_mapping = agts1_apron["suggested_mapping"]
    assert agts1_apron["source_facts"]["source_author"] == "AGTS1"
    assert agts1_apron["source_facts"]["image_count"] == 1201
    assert agts1_apron_mapping["local_class_to_source_labels"]["apron"] == ["apron"]
    assert "noapron" in agts1_apron_mapping["hard_negative_source_labels"]
    assert any("duplicate lineage against AGTS1 newapron" in item["implication"] for item in agts1_apron["source_research_evidence"])
    noisy_harness = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_harness_detection"
    )
    noisy_harness_mapping = noisy_harness["suggested_mapping"]
    assert noisy_harness_mapping["non_approving"] is True
    assert noisy_harness_mapping["local_class_to_source_labels"]["person"] == ["Person", "Worker", "person"]
    assert "safety-harness" in noisy_harness_mapping["local_class_to_source_labels"]["safety_harness"]
    assert "Lifeline" in noisy_harness_mapping["local_class_to_source_labels"]["safety_lanyard"]
    assert "no-safety-belt" in noisy_harness_mapping["hard_negative_source_labels"]
    assert noisy_harness["source_facts"]["image_count"] == 1431
    assert noisy_harness["source_facts"]["views"] == 1
    assert "safety-harness" in noisy_harness["source_facts"]["relevant_classes"]
    assert any(
        "64-class taxonomy" in item["finding"]
        for item in noisy_harness["source_research_evidence"]
    )
    assert any(
        "class list is noisy" in item["implication"]
        for item in noisy_harness["source_research_evidence"]
    )
    assert any(
        "default hosted workflow is not a reviewed harness detector" in item["implication"]
        for item in noisy_harness["source_research_evidence"]
    )
    scaffold = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_scaffold_harness"
    )
    scaffold_mapping = scaffold["suggested_mapping"]
    assert scaffold_mapping["non_approving"] is True
    assert scaffold_mapping["local_class_to_source_labels"]["person"] == ["human"]
    assert scaffold_mapping["local_class_to_source_labels"]["safety_harness"] == ["safety-harness"]
    assert "Guardrail" in scaffold_mapping["ignored_or_unmapped_source_labels"]
    assert "Suspension_scaffold" in scaffold_mapping["ignored_or_unmapped_source_labels"]
    assert scaffold["source_facts"]["views"] == 998
    assert scaffold["source_facts"]["downloads"] == 30
    assert any(
        "3,214 images" in item["finding"]
        for item in scaffold["source_research_evidence"]
    )
    assert any(
        "Guardrail" in item["finding"] and "Suspension_scaffold" in item["finding"]
        for item in scaffold["source_research_evidence"]
    )
    public_domain_harness = next(
        candidate
        for candidate in report["candidates"]
        if candidate["source_ref"] == "roboflow_safety_harness_public_domain"
    )
    assert public_domain_harness["source_facts"]["page_kind"] == "model"
    assert public_domain_harness["source_facts"]["model_dataset_image_count"] == 445
    public_domain_mapping = public_domain_harness["suggested_mapping"]
    assert public_domain_mapping["non_approving"] is True
    assert public_domain_mapping["local_class_to_source_labels"]["person"] == ["body"]
    assert public_domain_mapping["local_class_to_source_labels"]["safety_harness"] == ["harness"]
    assert public_domain_harness["source_facts"]["downloads"] == 174
    assert any(
        "Public Domain" in item["finding"] and "187 visible images" in item["finding"]
        for item in public_domain_harness["source_research_evidence"]
    )
    assert any(
        "safety-harness/3" in item["finding"]
        for item in public_domain_harness["source_research_evidence"]
    )
    assert any(
        "Public Domain Mark" in item["finding"]
        for item in public_domain_harness["source_research_evidence"]
    )
    harness_myproj = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_harness_myproj"
    )
    harness_myproj_mapping = harness_myproj["suggested_mapping"]
    assert harness_myproj["source_facts"]["source_author"] == "myproj"
    assert harness_myproj["source_facts"]["image_count"] == 1290
    assert harness_myproj_mapping["local_class_to_source_labels"]["safety_harness"] == ["Harness"]
    assert "NO-Safety Vest" in harness_myproj_mapping["hard_negative_source_labels"]
    assert "Safety Vest" in harness_myproj_mapping["hard_negative_source_labels"]
    assert any("vest-vs-harness hard-negative" in item["implication"] for item in harness_myproj["source_research_evidence"])
    public_plan = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_public_plan_harness"
    )
    public_plan_mapping = public_plan["suggested_mapping"]
    assert public_plan["source_facts"]["source_author"] == "PUBLIC PLAN"
    assert public_plan["source_facts"]["model_dataset_image_count"] == 1500
    assert public_plan["source_facts"]["page_metrics"]["map50"] == "98.8%"
    assert public_plan_mapping["local_class_to_source_labels"]["person"] == ["person"]
    assert public_plan_mapping["local_class_to_source_labels"]["safety_harness"] == ["safety-harness"]
    assert any(
        "RF-DETR hosted inference is not the target Jetson Nano local closed-set YOLO path" in item["implication"]
        for item in public_plan["source_research_evidence"]
    )
    yolo_lanyard = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_yolo_harness_lanyard"
    )
    yolo_lanyard_mapping = yolo_lanyard["suggested_mapping"]
    assert yolo_lanyard["source_facts"]["source_author"] == "yolo"
    assert yolo_lanyard["source_facts"]["image_count"] == 50
    assert yolo_lanyard_mapping["local_class_to_source_labels"]["person"] == ["person"]
    assert yolo_lanyard_mapping["local_class_to_source_labels"]["safety_harness"] == ["Harness", "harness"]
    assert yolo_lanyard_mapping["local_class_to_source_labels"]["safety_lanyard"] == [
        "lanyard-with-shock-basorber",
        "lifeline",
    ]
    assert "No harness" in yolo_lanyard_mapping["hard_negative_source_labels"]
    assert any("useful for reviewing lanyard/lifeline taxonomy" in item["implication"] for item in yolo_lanyard["source_research_evidence"])
    ppe_and_harness = next(
        candidate
        for candidate in report["candidates"]
        if candidate["source_ref"] == "roboflow_ppe_and_harness_wxfff"
    )
    ppe_and_harness_mapping = ppe_and_harness["suggested_mapping"]
    assert ppe_and_harness["source_facts"]["class_count"] == 26
    assert "Full-body-harness" in ppe_and_harness["source_facts"]["classes"]
    assert ppe_and_harness_mapping["local_class_to_source_labels"]["person"] == ["Person"]
    assert ppe_and_harness_mapping["local_class_to_source_labels"]["safety_harness"] == [
        "Full-body-harness"
    ]
    assert ppe_and_harness_mapping["local_class_to_source_labels"]["safety_lanyard"] == [
        "Lanyard-with-shock-basorber"
    ]
    assert any("Class list is noisy" in item["implication"] for item in ppe_and_harness["source_research_evidence"])
    safety_harness_dataset = next(
        candidate
        for candidate in report["candidates"]
        if candidate["source_ref"] == "roboflow_safety_harness_dataset"
    )
    safety_harness_dataset_mapping = safety_harness_dataset["suggested_mapping"]
    assert safety_harness_dataset["source_facts"]["source_author"] == "parkhm"
    assert safety_harness_dataset["source_facts"]["image_count"] == 65
    assert safety_harness_dataset_mapping["local_class_to_source_labels"]["safety_harness"] == [
        "safety-harness"
    ]
    assert "top" in safety_harness_dataset_mapping["ignored_or_unmapped_source_labels"]
    assert any("Very small safety-harness source" in item["implication"] for item in safety_harness_dataset["source_research_evidence"])
    harness_uvoia = next(
        candidate
        for candidate in report["candidates"]
        if candidate["source_ref"] == "roboflow_harness_uvoia_public_domain"
    )
    harness_uvoia_mapping = harness_uvoia["suggested_mapping"]
    assert harness_uvoia["source_facts"]["source_author"] == "Safety Harness Dataset"
    assert harness_uvoia["source_facts"]["model_dataset_image_count"] == 233
    assert harness_uvoia["source_facts"]["classes"] == ["0", "1"]
    assert harness_uvoia_mapping["local_class_to_source_labels"] == {}
    assert harness_uvoia_mapping["ignored_or_unmapped_source_labels"] == ["0", "1"]
    assert any("numeric 0/1 classes are ambiguous" in item["implication"] for item in harness_uvoia["source_research_evidence"])
    full_body_harness = next(
        candidate for candidate in report["candidates"] if candidate["source_ref"] == "roboflow_full_body_harness"
    )
    full_body_mapping = full_body_harness["suggested_mapping"]
    assert full_body_harness["source_facts"]["source_author"] == "labellingapd"
    assert full_body_harness["source_facts"]["image_count"] == 70
    assert full_body_mapping["local_class_to_source_labels"]["safety_harness"] == [
        "full-body-harness"
    ]
    assert any("Very small full-body-harness source" in item["implication"] for item in full_body_harness["source_research_evidence"])
    legal_references = work_at_height["legal_references"]
    assert legal_references["license_reference_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert legal_references["platform_references"]["platform_terms_url"] == "https://roboflow.com/terms"
    assert legal_references["platform_references"]["export_docs_url"] == (
        "https://docs.roboflow.com/datasets/dataset-versions/exporting-data"
    )
    assert legal_references["platform_references"]["universe_license_docs_url"] == (
        "https://docs.roboflow.com/universe/find-a-dataset-on-universe"
    )
    assert legal_references["platform_references"]["universe_download_docs_url"] == (
        "https://docs.roboflow.com/universe/download-a-universe-dataset"
    )
    rendered = out.read_text(encoding="utf-8")
    assert "Apron/Harness Public Seed Source Review" in rendered
    assert "Priority Review Queue" in rendered
    assert "`harness_required` | `roboflow_work_at_height_safety` (10)" in rendered
    assert "`apron_required` | `roboflow_workspace_otd88_fjwepfj1` (14)" in rendered
    assert "roboflow_safety_harness_dataset" in rendered
    assert "Review Focus" in rendered
    assert "manifest_import_plan" in rendered
    assert import_template["kind"] == "apron_harness_seed_import_manifest"
    assert import_template["source_review_report"] == "qa/video_eval/results/apron_harness_seed_source_review.json"
    fill_contract = import_template["fill_contract"]
    assert "This template is not approval" in fill_contract["approval_boundary"]
    assert "raw_export_local_path points to the reviewed local YOLO export ZIP" in fill_contract[
        "required_before_include_in_training"
    ]
    assert "review_status=approved_for_training" in fill_contract[
        "required_before_include_in_training"
    ]
    assert "review_status=approved" not in fill_contract["required_before_include_in_training"]
    assert "include_in_training=true" in fill_contract["forbidden_until_approved"]
    assert any("--validate-import-manifest" in command for command in fill_contract["validation_commands"])
    assert len(import_template["imports"]) == review_queue["reviewable_count"]
    assert not any(item["source_ref"] == "roboflow_harness_detection_v1" for item in import_template["imports"])
    assert all(item["include_in_training"] is False for item in import_template["imports"])
    minimum_refs = doctor.minimum_approval_source_refs(report)
    assert minimum_refs == {
        "roboflow_work_at_height_safety",
        "roboflow_harness_s4xxh",
        "roboflow_workspace_otd88_fjwepfj1",
    }
    minimum_template = doctor.build_import_manifest_template(
        report,
        seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
        source_refs=minimum_refs,
        template_scope="minimum_priority_coverage_sources",
    )
    assert minimum_template["template_scope"] == "minimum_priority_coverage_sources"
    assert minimum_template["selected_source_refs"] == sorted(minimum_refs)
    assert [item["source_ref"] for item in minimum_template["imports"]] == [
        "roboflow_work_at_height_safety",
        "roboflow_harness_s4xxh",
        "roboflow_workspace_otd88_fjwepfj1",
    ]
    assert all(item["include_in_training"] is False for item in minimum_template["imports"])
    assert all(item["expected_labeled_images_per_class"]["person"] == 0 for item in minimum_template["imports"])
    assert "completed_review" in import_template["imports"][0]
    assert import_template["imports"][0]["review_priority"] == 10
    assert import_template["imports"][0]["source_ref"] == "roboflow_work_at_height_safety"
    assert import_template["imports"][0]["review_focus"]
    assert import_template["imports"][0]["review_artifacts"]["review_packet_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_packet.md"
    )
    assert import_template["imports"][0]["review_artifacts"]["review_evidence_template_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_evidence.yaml"
    )
    assert "approved_for_training=true" in " ".join(import_template["imports"][0]["approval_preconditions"])
    assert import_template["imports"][0]["class_mapping"] == {}
    assert import_template["imports"][0]["suggested_mapping"]["non_approving"] is True
    assert import_template["imports"][0]["suggested_mapping"]["local_class_to_source_labels"]["person"] == ["person"]
    fill_plan = import_template["imports"][0]["seed_import_fill_plan"]
    assert fill_plan["non_approving"] is True
    assert fill_plan["required_local_classes"] == ["person", "safety_harness", "safety_lanyard"]
    assert fill_plan["reviewed_class_mapping_starter"] == {
        "person": ["person"],
        "safety_harness": ["harness"],
    }
    assert fill_plan["missing_required_classes_from_suggestion"] == ["safety_lanyard"]
    assert fill_plan["expected_count_classes_that_must_be_nonzero"] == [
        "person",
        "safety_harness",
        "safety_lanyard",
    ]
    assert "raw_export_local_path" in fill_plan["required_fields_before_include_in_training"]
    assert len(checklist_rows) == review_queue["reviewable_count"]
    assert not any(row["source_ref"] == "roboflow_harness_detection_v1" for row in checklist_rows)
    assert checklist_rows[0]["review_priority"] == "10"
    assert checklist_rows[0]["source_ref"] == "roboflow_work_at_height_safety"
    assert checklist_rows[0]["approved_for_training"] == "false"
    assert checklist_rows[0]["training_usable"] == "false"
    assert checklist_rows[0]["review_packet_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_packet.md"
    )
    assert len(checklist_rows[0]["review_packet_sha256"]) == 64
    assert checklist_rows[0]["review_evidence_template_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_evidence.yaml"
    )
    assert len(checklist_rows[0]["review_evidence_template_sha256"]) == 64
    assert checklist_rows[0]["review_evidence_path"] == ""
    assert checklist_rows[0]["review_evidence_sha256"] == ""
    assert checklist_rows[0]["license_terms"] == "false"
    assert checklist_rows[0]["manifest_import_plan"] == "false"
    assert checklist_rows[0]["review_focus"]
    assert len(written_templates) == review_queue["reviewable_count"]
    assert not any(item["source_ref"] == "roboflow_harness_detection_v1" for item in written_templates)
    first_template = yaml.safe_load((ROOT / written_templates[0]["path"]).read_text(encoding="utf-8"))
    assert first_template["kind"] == doctor.REVIEW_EVIDENCE_KIND
    assert first_template["version"] == 1
    assert first_template["source_ref"] == "roboflow_work_at_height_safety"
    assert first_template["legal_references"]["license_reference_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert first_template["legal_references"]["platform_references"]["platform_terms_url"] == "https://roboflow.com/terms"
    assert first_template["legal_references"]["platform_references"]["universe_license_docs_url"] == (
        "https://docs.roboflow.com/universe/find-a-dataset-on-universe"
    )
    assert first_template["source_facts"]["task"] == "object_detection"
    assert first_template["source_facts"]["public_page_image_count"] == 12805
    assert first_template["suggested_mapping"]["non_approving"] is True
    assert first_template["suggested_mapping"]["local_class_to_source_labels"]["safety_harness"] == ["harness"]
    assert first_template["agent_collected_review_hints"]["license_terms"]["evidence_refs"]
    assert "https://roboflow.com/terms" in (
        first_template["agent_collected_review_hints"]["license_terms"]["evidence_refs"]
    )
    assert any(
        "creativecommons.org/licenses/by/4.0" in ref
        for ref in first_template["agent_collected_review_hints"]["license_terms"]["evidence_refs"]
    )
    assert any(
        "dataset-versions/exporting-data" in ref
        for ref in first_template["agent_collected_review_hints"]["export_terms"]["evidence_refs"]
    )
    assert "safety_harness" in first_template["agent_collected_review_hints"]["class_mapping"]["notes"]
    assert set(first_template["review_items"]) == doctor.REQUIRED_REVIEW_ITEMS
    assert all(item["approved"] is False for item in first_template["review_items"].values())
    assert all(item["prefilled_by_agent"] is True for item in first_template["review_items"].values())
    assert all(item["evidence_ref"] for item in first_template["review_items"].values())
    assert "Agent prefill only" in first_template["review_items"]["license_terms"]["notes"]
    assert "commercial training rights" in first_template["review_items"]["license_terms"]["notes"]
    assert "dataset-versions/exporting-data" in first_template["review_items"]["export_terms"]["evidence_ref"]
    assert "safety_lanyard" in first_template["review_items"]["hard_negative_coverage"]["notes"]


def test_seed_source_review_evidence_template_writer_uses_stable_filenames(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    evidence_dir.mkdir()
    stale_file = evidence_dir / "roboflow_harness_detection_v1__harness_required.review_evidence.yaml"
    stale_file.write_text("stale: true\n", encoding="utf-8")

    written = doctor.write_review_evidence_templates(report, evidence_dir)

    assert len(written) == report["review_queue_summary"]["reviewable_count"]
    assert all(Path(item["path"]).suffix == ".yaml" for item in written)
    assert all((ROOT / item["path"]).exists() for item in written)
    assert not any(item["source_ref"] == "roboflow_harness_detection_v1" for item in written)
    assert not stale_file.exists()
    assert any(
        "roboflow_work_at_height_safety__harness_required.review_evidence.yaml"
        in item["path"]
        for item in written
    )
    template = yaml.safe_load(
        (
            tmp_path
            / "evidence_templates"
            / "roboflow_work_at_height_safety__harness_required.review_evidence.yaml"
        ).read_text(encoding="utf-8")
    )
    assert template["source_facts"]["dataset_version"]["id"] == "work-at-height-safety/3"
    assert template["agent_collected_review_hints"]["train_val_test_split"]["evidence_refs"]
    assert any("YOLO-family export options" in item["finding"] for item in template["source_research_evidence"])
    assert any("Public Plan service use" in item["finding"] for item in template["source_research_evidence"])
    assert any("CC BY 4.0 allows commercial" in item["finding"] for item in template["source_research_evidence"])
    assert template["review_items"]["train_val_test_split"]["prefilled_by_agent"] is True
    assert "work-at-height-safety/dataset/3" in template["review_items"]["train_val_test_split"]["evidence_ref"]
    assert "Observed source split" in template["review_items"]["train_val_test_split"]["notes"]
    assert template["agent_research_boundary"]["can_collect_evidence"] is True
    assert template["agent_research_boundary"]["can_approve_training"] is False
    assert any(
        "Open the source page" in item
        for item in template["agent_research_boundary"]["agent_research_tasks"]
    )
    assert any(
        "Confirm commercial training rights" in item
        for item in template["agent_research_boundary"]["human_approval_tasks"]
    )


def test_seed_source_review_packet_writer_uses_stable_filenames(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    packet_dir.mkdir()
    stale_file = packet_dir / "roboflow_harness_detection_v1__harness_required.review_packet.md"
    stale_file.write_text("# Stale\n", encoding="utf-8")
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }

    written = doctor.write_review_packets(report, packet_dir)

    assert len(written) == report["review_queue_summary"]["reviewable_count"]
    assert all(item["path"].endswith(".review_packet.md") for item in written)
    assert all((ROOT / item["path"]).exists() for item in written)
    assert not any(item["source_ref"] == "roboflow_harness_detection_v1" for item in written)
    assert not stale_file.exists()
    work_at_height = next(
        item
        for item in written
        if "roboflow_work_at_height_safety__harness_required.review_packet.md" in item["path"]
    )
    packet = (ROOT / work_at_height["path"]).read_text(encoding="utf-8")
    assert "## Agent Research vs Human Approval" in packet
    assert "Agent/browser research may gather source facts" in packet
    assert "must not set `approved_for_training=true`" in packet
    assert "Human/legal approval tasks" in packet
    assert "Confirm commercial training rights" in packet
    assert "## Source Facts" in packet
    assert "### Observed Source Facts" in packet
    assert "dataset_version" in packet
    assert "### Current Source-Research Evidence" in packet
    assert "YOLO-family export options" in packet
    assert "Require provenance review before accepting the CC BY 4.0 page label" in packet
    assert "public_page_image_count" in packet
    assert "## Legal References" in packet
    assert "https://roboflow.com/terms" in packet
    assert "https://docs.roboflow.com/datasets/dataset-versions/exporting-data" in packet
    assert "https://docs.roboflow.com/universe/find-a-dataset-on-universe" in packet
    assert "https://docs.roboflow.com/universe/download-a-universe-dataset" in packet
    assert "https://creativecommons.org/licenses/by/4.0/" in packet
    assert "## Suggested Class Mapping" in packet
    assert "`safety_harness`" in packet
    assert "hard_negative" in packet
    assert "## Required Review Evidence" in packet
    assert "roboflow_work_at_height_safety__harness_required.review_evidence.yaml" in packet
    assert "prefill `review_items.*.evidence_ref` with agent-collected hints" in packet
    assert "prefilled_by_agent: true" in packet
    assert "`raw_export_local_path` must point to the reviewed local YOLO export ZIP" in packet
    assert "Required local classes" in packet
    assert "safety_harness" in packet
    assert "safety_lanyard" in packet


def test_seed_source_review_queue_joins_generated_artifacts(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {
        "path": str(checklist_path),
        "sha256": _sha256_file(checklist_path),
    }
    report["seed_import_manifest_template"] = {
        "path": str(import_template_path),
        "sha256": _sha256_file(import_template_path),
    }

    queue = doctor.build_review_queue(report)

    assert len(queue) == report["candidate_count"]
    first = queue[0]
    assert first["source_ref"] == "roboflow_work_at_height_safety"
    assert first["review_artifacts"]["review_packet_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_packet.md"
    )
    assert first["review_artifacts"]["review_evidence_template_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_evidence.yaml"
    )
    assert first["review_artifacts"]["review_checklist_csv_path"] == str(checklist_path)
    assert first["review_artifacts"]["review_checklist_csv_sha256"] == _sha256_file(checklist_path)
    assert first["review_artifacts"]["seed_import_manifest_template_path"] == str(import_template_path)
    assert first["review_artifacts"]["seed_import_manifest_template_sha256"] == _sha256_file(
        import_template_path
    )
    unavailable = next(item for item in queue if item["source_ref"] == "roboflow_harness_detection_v1")
    assert unavailable["approval_status"] == "rejected"
    assert unavailable["next_action"] == "keep_source_blocked_or_replace_with_verifiable_source"
    assert unavailable["review_artifacts"]["review_packet_path"] == ""
    assert unavailable["review_artifacts"]["review_evidence_template_path"] == ""


def test_seed_source_next_review_batch_is_non_approving_agent_handoff(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    minimum_import_template_path = tmp_path / "apron_harness_minimum_seed_import_manifest.template.yaml"
    batch_path = tmp_path / "apron_harness_next_source_review_batch.json"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    minimum_import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
                source_refs=doctor.minimum_approval_source_refs(report),
                template_scope="minimum_priority_coverage_sources",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {
        "path": str(checklist_path),
        "sha256": _sha256_file(checklist_path),
    }
    report["seed_import_manifest_template"] = {
        "path": str(import_template_path),
        "sha256": _sha256_file(import_template_path),
    }
    report["minimum_seed_import_manifest_template"] = {
        "path": str(minimum_import_template_path),
        "sha256": _sha256_file(minimum_import_template_path),
    }
    report["review_queue"] = doctor.build_review_queue(report)

    metadata = doctor.write_next_review_batch(
        report,
        batch_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["next_review_batch"] = metadata
    batch = json.loads(batch_path.read_text(encoding="utf-8"))

    assert metadata["selected_count"] == 3
    assert batch["kind"] == doctor.NEXT_REVIEW_BATCH_KIND
    assert batch["source_review_report"] == "qa/video_eval/results/apron_harness_seed_source_review.json"
    assert batch["source_review_sha256"] == doctor.source_review_fingerprint(report)
    validation = doctor.validate_next_review_batch(batch_path, report)
    assert validation["ok"] is True
    assert validation["minimum_review_path_count"] == 3
    assert validation["source_review_sha256_matches"] is True
    assert batch["approval_guardrail"]["agent_can_collect_evidence"] is True
    assert batch["approval_guardrail"]["agent_can_approve_training"] is False
    assert "include_in_training=true" in batch["approval_guardrail"]["forbidden_fields"]
    assert "review_items.*.approved=true" in batch["approval_guardrail"]["forbidden_fields"]
    assert batch["minimum_review_path"]["available"] is True
    assert batch["minimum_review_path"]["source_refs"] == [
        "roboflow_workspace_otd88_fjwepfj1",
        "roboflow_work_at_height_safety",
        "roboflow_harness_s4xxh",
    ]
    assert batch["minimum_review_path"]["source_count"] == 3
    assert "does not approve training" in batch["minimum_review_path"]["approval_boundary"]
    assert batch["minimum_review_path"]["items"][0]["seed_import_manifest_template_path"] == (
        "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml"
    )
    assert [item["source_ref"] for item in batch["items"]] == [
        "roboflow_work_at_height_safety",
        "roboflow_harness_s4xxh",
        "roboflow_workspace_otd88_fjwepfj1",
    ]
    first = batch["items"][0]
    assert first["rank"] == 1
    assert first["training_usable"] is False
    assert first["review_artifacts"]["review_packet_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_packet.md"
    )
    assert first["review_artifacts"]["review_evidence_template_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_evidence.yaml"
    )
    assert first["review_artifacts"]["review_prefill_path"].endswith(
        "apron_harness_work_at_height_safety_review_prefill_2026_06_23.md"
    )
    assert first["review_packet_path"] == first["review_artifacts"]["review_packet_path"]
    assert first["review_packet_sha256"] == first["review_artifacts"]["review_packet_sha256"]
    assert first["review_evidence_template_path"] == (
        first["review_artifacts"]["review_evidence_template_path"]
    )
    assert first["review_evidence_template_sha256"] == (
        first["review_artifacts"]["review_evidence_template_sha256"]
    )
    assert first["review_prefill_path"] == first["review_artifacts"]["review_prefill_path"]
    assert len(first["review_prefill_sha256"]) == 64
    assert first["review_prefill_sha256"] == first["review_artifacts"]["review_prefill_sha256"]
    assert first["review_checklist_csv_path"] == first["review_artifacts"]["review_checklist_csv_path"]
    assert first["review_checklist_csv_sha256"] == first["review_artifacts"]["review_checklist_csv_sha256"]
    assert first["seed_import_manifest_template_path"] == (
        first["review_artifacts"]["seed_import_manifest_template_path"]
    )
    assert first["seed_import_manifest_template_sha256"] == (
        first["review_artifacts"]["seed_import_manifest_template_sha256"]
    )
    assert first["review_artifacts"]["review_checklist_csv_path"] == str(checklist_path)
    assert first["review_artifacts"]["seed_import_manifest_template_path"] == str(import_template_path)
    assert first["source_research_status"]["status"] == "evidence_ready_for_human_review"
    assert first["source_research_status"]["evidence_ready_for_human_review"] is True
    assert first["source_research_status"]["evidence_count"] >= doctor.SOURCE_RESEARCH_EVIDENCE_READY_MIN_ITEMS
    assert first["seed_import_fill_plan"]["non_approving"] is True
    assert first["seed_import_fill_plan"]["required_local_classes"] == [
        "person",
        "safety_harness",
        "safety_lanyard",
    ]
    assert first["seed_import_fill_plan"]["missing_required_classes_from_suggestion"] == [
        "safety_lanyard"
    ]
    assert "review_status=approved_for_training" in first["seed_import_fill_plan"][
        "required_fields_before_include_in_training"
    ]
    assert any("Open the source page" in task for task in first["agent_research_tasks"])
    assert any("Confirm commercial training rights" in task for task in first["human_approval_tasks"])


def test_seed_source_next_review_batch_rejects_stale_minimum_path(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    batch_path = tmp_path / "apron_harness_next_source_review_batch.json"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {"path": str(checklist_path), "sha256": _sha256_file(checklist_path)}
    report["seed_import_manifest_template"] = {
        "path": str(import_template_path),
        "sha256": _sha256_file(import_template_path),
    }
    report["review_queue"] = doctor.build_review_queue(report)
    doctor.write_next_review_batch(
        report,
        batch_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["minimum_review_path"]["source_refs"][0] = "wrong_source"
    batch_path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")

    validation = doctor.validate_next_review_batch(batch_path, report)

    assert validation["ok"] is False
    assert any("minimum_review_path.source_refs must match" in error for error in validation["errors"])


def test_seed_source_review_kickoff_is_non_approving_markdown(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    kickoff_path = tmp_path / "apron_harness_source_review_kickoff.md"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {
        "path": str(checklist_path),
        "sha256": _sha256_file(checklist_path),
    }
    report["seed_import_manifest_template"] = {
        "path": str(import_template_path),
        "sha256": _sha256_file(import_template_path),
    }
    report["review_queue"] = doctor.build_review_queue(report)

    metadata = doctor.write_review_kickoff(
        report,
        kickoff_path,
        limit=4,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    rendered = kickoff_path.read_text(encoding="utf-8")
    report["review_kickoff"] = metadata

    assert metadata["kind"] == doctor.REVIEW_KICKOFF_KIND
    assert metadata["selected_count"] == 4
    assert metadata["source_review_sha256"] == doctor.source_review_fingerprint(report)
    assert "# Apron/Harness Source Review Kickoff" in rendered
    assert "non-approving operator handoff" in rendered
    assert "Agents must not set approval" in rendered
    assert "include_in_training=true" in rendered
    assert "Source review bundle: `qa/video_eval/results/apron_harness_source_review_bundle.json`" in rendered
    assert "roboflow_work_at_height_safety" in rendered
    assert "https://universe.roboflow.com/proyecto-prevencion-predictiva/work-at-height-safety" in rendered
    assert "Roboflow Universe page lists CC BY 4.0." in rendered
    assert "roboflow_safety_food_system" in rendered
    assert "## Source Coverage Snapshot" in rendered
    assert "non-approving suggested mappings" in rendered
    assert "| `apron_required` | none | `candidate_person_mapping_present_pending_review` | `candidate_coverage_complete_pending_review` | roboflow_workspace_otd88_fjwepfj1 |" in rendered
    assert "| `harness_required` | none | `candidate_person_mapping_present_pending_review` | `candidate_coverage_complete_pending_review` | roboflow_work_at_height_safety, roboflow_harness_s4xxh |" in rendered
    assert "## Minimum Review Path" in rendered
    assert "shortest non-approving path to apron/harness coverage" in rendered
    assert "| `roboflow_workspace_otd88_fjwepfj1` | `apron_required` |" in rendered
    assert "| `roboflow_work_at_height_safety` | `harness_required` |" in rendered
    assert "| `roboflow_harness_s4xxh` | `harness_required` |" in rendered
    assert "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in rendered
    assert "review_packet.md" in rendered
    assert "review_evidence.yaml" in rendered
    assert "Prefill Memo" in rendered
    assert "apron_harness_safety_food_system_review_prefill_2026_06_23.md" in rendered
    assert "## Seed Import Fill Plan" in rendered
    assert (
        "| 1 | `roboflow_work_at_height_safety` | person, safety_harness, safety_lanyard | "
        "person=person; safety_harness=harness | safety_lanyard | person, safety_harness, safety_lanyard | "
        "review_status=approved_for_training, reviewed_by, reviewed_at, manifest_import_path, "
        "raw_export_ref, raw_export_sha256, +8 more |"
    ) in rendered
    assert "## YOLO Export ZIP Proof" in rendered
    assert "A source approval is not enough to import seed data" in rendered
    assert "`raw_export_ref` is a remote immutable export reference" in rendered
    assert "`raw_export_sha256` matches the reviewed local ZIP at `raw_export_local_path`" in rendered
    assert "contains `data.yaml` plus train/valid/test image and label folders" in rendered
    assert "orphan labels block import" in rendered
    assert "`class_mapping` covers every required local class" in rendered
    assert "expected_count_classes_that_must_be_nonzero" in rendered
    assert "Review artifact paths and SHA-256 values still match" in rendered
    assert "Validate the next-review batch and source-review bundle hashes before using generated packets" in rendered
    assert "--review-bundle-out \"\"" not in rendered
    assert "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json" in rendered
    assert "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in rendered
    assert "--apply-review-checklist-csv" in rendered
    assert "--validate-import-manifest" in rendered
    assert "IMPORT_MANIFEST: gate=pass" in rendered
    assert "IMPORT_MANIFEST: gate=blocked" in rendered
    assert "exit nonzero" in rendered
    assert "closed-set YOLO26n/s training" in rendered


def test_seed_source_review_bundle_manifest_records_handoff_artifacts(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    minimum_import_template_path = tmp_path / "apron_harness_minimum_seed_import_manifest.template.yaml"
    batch_path = tmp_path / "apron_harness_next_source_review_batch.json"
    kickoff_path = tmp_path / "apron_harness_source_review_kickoff.md"
    work_order_path = tmp_path / "apron_harness_seed_source_review.md"
    coverage_plan_path = tmp_path / "apron_harness_source_coverage_plan.json"
    bundle_path = tmp_path / "apron_harness_source_review_bundle.json"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    minimum_import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
                source_refs=doctor.minimum_approval_source_refs(report),
                template_scope="minimum_priority_coverage_sources",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {
        "path": str(checklist_path),
        "sha256": _sha256_file(checklist_path),
    }
    report["seed_import_manifest_template"] = {
        "path": str(import_template_path),
        "sha256": _sha256_file(import_template_path),
    }
    report["minimum_seed_import_manifest_template"] = {
        "path": str(minimum_import_template_path),
        "sha256": _sha256_file(minimum_import_template_path),
    }
    report["review_queue"] = doctor.build_review_queue(report)
    report["next_review_batch"] = doctor.write_next_review_batch(
        report,
        batch_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["review_kickoff"] = doctor.write_review_kickoff(
        report,
        kickoff_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["source_coverage_plan"] = doctor.write_source_coverage_plan(
        report,
        coverage_plan_path,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    assert report["source_coverage_plan"]["person_box_reconciliation"]["apron_required"]["status"] == (
        "candidate_person_mapping_present_pending_review"
    )
    assert report["source_coverage_plan"]["person_box_reconciliation"]["harness_required"]["status"] == (
        "candidate_person_mapping_present_pending_review"
    )
    work_order_path.write_text(doctor.render_work_order(report), encoding="utf-8")
    report["work_order"] = {
        "path": str(work_order_path),
        "sha256": _sha256_file(work_order_path),
    }

    metadata = doctor.write_review_bundle_manifest(
        report,
        bundle_path,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["review_bundle"] = metadata
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert metadata["kind"] == doctor.REVIEW_BUNDLE_KIND
    assert metadata["review_packet_count"] == len(report["review_packets"]["files"])
    assert metadata["review_evidence_template_count"] == len(report["review_evidence_templates"]["files"])
    assert metadata["review_prefill_count"] == 5
    assert metadata["source_review_sha256"] == doctor.source_review_fingerprint(report)
    assert bundle["kind"] == doctor.REVIEW_BUNDLE_KIND
    assert bundle["source_review_report"] == "qa/video_eval/results/apron_harness_seed_source_review.json"
    assert bundle["source_review_gate_passed"] is False
    assert bundle["approval_guardrail"]["agent_can_approve_training"] is False
    assert "include_in_training=true" in bundle["approval_guardrail"]["forbidden_fields"]
    assert bundle["minimum_review_path"]["source_refs"] == [
        "roboflow_workspace_otd88_fjwepfj1",
        "roboflow_work_at_height_safety",
        "roboflow_harness_s4xxh",
    ]
    assert bundle["minimum_review_path"]["source_count"] == 3
    assert bundle["minimum_review_path"]["items"][0]["review_packet_path"].endswith(
        "roboflow_workspace_otd88_fjwepfj1__apron_required.review_packet.md"
    )
    single_files = {item["kind"]: item for item in bundle["artifacts"]["single_files"]}
    assert single_files["source_review_work_order"]["sha256"] == _sha256_file(work_order_path)
    assert single_files["review_checklist_csv"]["sha256"] == _sha256_file(checklist_path)
    assert single_files["seed_import_manifest_template"]["sha256"] == _sha256_file(import_template_path)
    assert single_files["minimum_seed_import_manifest_template"]["sha256"] == _sha256_file(
        minimum_import_template_path
    )
    assert single_files["next_review_batch"]["sha256"] == _sha256_file(batch_path)
    assert single_files["review_kickoff"]["sha256"] == _sha256_file(kickoff_path)
    assert single_files["source_coverage_plan"]["sha256"] == _sha256_file(coverage_plan_path)
    assert single_files["source_recheck"]["path"] == (
        "qa/video_eval/results/apron_harness_source_recheck_2026_06_24.md"
    )
    assert len(single_files["source_recheck"]["sha256"]) == 64
    assert bundle["artifacts"]["review_packets"]["count"] == len(report["review_packets"]["files"])
    assert bundle["artifacts"]["review_evidence_templates"]["count"] == len(
        report["review_evidence_templates"]["files"]
    )
    assert bundle["artifacts"]["review_prefills"]["count"] == 5
    assert any(
        item["path"].endswith("apron_harness_safety_food_system_review_prefill_2026_06_23.md")
        for item in bundle["artifacts"]["review_prefills"]["files"]
    )
    assert any("--review-bundle-out" in command for command in bundle["verification_commands"])
    assert all('--review-bundle-out ""' not in command for command in bundle["verification_commands"])
    assert any("--minimum-import-template-out" in command for command in bundle["verification_commands"])
    assert any("--source-coverage-plan-out" in command for command in bundle["verification_commands"])
    assert any("--validate-review-bundle" in command for command in bundle["verification_commands"])
    assert any("--validate-import-manifest" in command for command in bundle["verification_commands"])


def test_seed_source_review_bundle_manifest_validates_handoff_hashes(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    batch_path = tmp_path / "apron_harness_next_source_review_batch.json"
    kickoff_path = tmp_path / "apron_harness_source_review_kickoff.md"
    work_order_path = tmp_path / "apron_harness_seed_source_review.md"
    bundle_path = tmp_path / "apron_harness_source_review_bundle.json"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {
        "path": str(checklist_path),
        "sha256": _sha256_file(checklist_path),
    }
    report["seed_import_manifest_template"] = {
        "path": str(import_template_path),
        "sha256": _sha256_file(import_template_path),
    }
    report["review_queue"] = doctor.build_review_queue(report)
    report["next_review_batch"] = doctor.write_next_review_batch(
        report,
        batch_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["review_kickoff"] = doctor.write_review_kickoff(
        report,
        kickoff_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    work_order_path.write_text(doctor.render_work_order(report), encoding="utf-8")
    report["work_order"] = {
        "path": str(work_order_path),
        "sha256": _sha256_file(work_order_path),
    }
    report["review_bundle"] = doctor.write_review_bundle_manifest(
        report,
        bundle_path,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )

    validation = doctor.validate_review_bundle_manifest(bundle_path, report)

    expected_prefill_count = sum(
        1
        for item in report["review_queue"]
        if item["review_artifacts"]["review_prefill_path"]
    )
    assert validation["ok"] is True
    assert validation["source_review_sha256_matches"] is True
    assert validation["checked_artifact_count"] == (
        6
        + len(report["review_packets"]["files"])
        + len(report["review_evidence_templates"]["files"])
        + expected_prefill_count
    )
    assert validation["review_packet_count"] == len(report["review_packets"]["files"])
    assert validation["review_evidence_template_count"] == len(report["review_evidence_templates"]["files"])
    assert validation["review_prefill_count"] == expected_prefill_count
    assert validation["minimum_review_path_count"] == 3


def test_seed_source_review_bundle_manifest_rejects_stale_handoff_artifact(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    batch_path = tmp_path / "apron_harness_next_source_review_batch.json"
    kickoff_path = tmp_path / "apron_harness_source_review_kickoff.md"
    work_order_path = tmp_path / "apron_harness_seed_source_review.md"
    bundle_path = tmp_path / "apron_harness_source_review_bundle.json"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {
        "path": str(checklist_path),
        "sha256": _sha256_file(checklist_path),
    }
    report["seed_import_manifest_template"] = {
        "path": str(import_template_path),
        "sha256": _sha256_file(import_template_path),
    }
    report["review_queue"] = doctor.build_review_queue(report)
    report["next_review_batch"] = doctor.write_next_review_batch(
        report,
        batch_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["review_kickoff"] = doctor.write_review_kickoff(
        report,
        kickoff_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    work_order_path.write_text(doctor.render_work_order(report), encoding="utf-8")
    report["work_order"] = {
        "path": str(work_order_path),
        "sha256": _sha256_file(work_order_path),
    }
    report["review_bundle"] = doctor.write_review_bundle_manifest(
        report,
        bundle_path,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    checklist_path.write_text("tampered checklist\n", encoding="utf-8")

    validation = doctor.validate_review_bundle_manifest(bundle_path, report)

    assert validation["ok"] is False
    assert any("sha256 does not match generated artifact file" in error for error in validation["errors"])


def test_seed_source_review_bundle_manifest_rejects_stale_minimum_review_path(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    evidence_dir = tmp_path / "evidence_templates"
    packet_dir = tmp_path / "review_packets"
    checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    minimum_import_template_path = tmp_path / "apron_harness_minimum_seed_import_manifest.template.yaml"
    batch_path = tmp_path / "apron_harness_next_source_review_batch.json"
    kickoff_path = tmp_path / "apron_harness_source_review_kickoff.md"
    work_order_path = tmp_path / "apron_harness_seed_source_review.md"
    coverage_plan_path = tmp_path / "apron_harness_source_coverage_plan.json"
    bundle_path = tmp_path / "apron_harness_source_review_bundle.json"
    report["review_evidence_templates"] = {
        "dir": str(evidence_dir),
        "files": doctor.write_review_evidence_templates(report, evidence_dir),
    }
    report["review_packets"] = {
        "dir": str(packet_dir),
        "files": doctor.write_review_packets(report, packet_dir),
    }
    checklist_path.write_text(doctor.build_review_checklist_csv(report), encoding="utf-8")
    import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    minimum_import_template_path.write_text(
        yaml.safe_dump(
            doctor.build_import_manifest_template(
                report,
                seed_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
                source_refs=doctor.minimum_approval_source_refs(report),
                template_scope="minimum_priority_coverage_sources",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["review_checklist_csv"] = {"path": str(checklist_path), "sha256": _sha256_file(checklist_path)}
    report["seed_import_manifest_template"] = {"path": str(import_template_path), "sha256": _sha256_file(import_template_path)}
    report["minimum_seed_import_manifest_template"] = {
        "path": str(minimum_import_template_path),
        "sha256": _sha256_file(minimum_import_template_path),
    }
    report["review_queue"] = doctor.build_review_queue(report)
    report["next_review_batch"] = doctor.write_next_review_batch(
        report,
        batch_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["review_kickoff"] = doctor.write_review_kickoff(
        report,
        kickoff_path,
        limit=3,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    report["source_coverage_plan"] = doctor.write_source_coverage_plan(
        report,
        coverage_plan_path,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    work_order_path.write_text(doctor.render_work_order(report), encoding="utf-8")
    report["work_order"] = {"path": str(work_order_path), "sha256": _sha256_file(work_order_path)}
    report["review_bundle"] = doctor.write_review_bundle_manifest(
        report,
        bundle_path,
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["minimum_review_path"]["source_refs"][0] = "wrong_source"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    validation = doctor.validate_review_bundle_manifest(bundle_path, report)

    assert validation["ok"] is False
    assert any("minimum_review_path.source_refs must match" in error for error in validation["errors"])


def test_seed_source_work_order_lists_review_evidence_templates(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    report["review_evidence_templates"] = {
        "dir": str(tmp_path / "evidence_templates"),
        "files": doctor.write_review_evidence_templates(report, tmp_path / "evidence_templates"),
    }
    report["review_packets"] = {
        "dir": str(tmp_path / "review_packets"),
        "files": doctor.write_review_packets(report, tmp_path / "review_packets"),
    }
    report["source_coverage_plan"] = doctor.write_source_coverage_plan(
        report,
        tmp_path / "apron_harness_source_coverage_plan.json",
        source_review_report="qa/video_eval/results/apron_harness_seed_source_review.json",
    )

    rendered = doctor.render_work_order(report)

    assert "## Source Research Readiness" in rendered
    assert "### Evidence-Ready Sources" in rendered
    assert "roboflow_work_at_height_safety" in rendered
    assert "### Research-Closed Sources" in rendered
    assert "source_unavailable_after_agent_research" in rendered
    assert "roboflow_harness_detection_v1" in rendered
    assert "### Next Agent Research Sources" in rendered
    assert "| n/a | none | n/a | none | human_legal_review_and_seed_import_manifest |" in rendered
    assert "## Review Evidence Templates" in rendered
    assert "roboflow_work_at_height_safety__harness_required.review_evidence.yaml" in rendered
    assert "## Review Packets" in rendered
    assert "roboflow_work_at_height_safety__harness_required.review_packet.md" in rendered
    assert "shasum -a 256" in rendered
    assert "## Source Review Kickoff" in rendered
    assert "--review-kickoff-out" in rendered
    assert "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json" in rendered
    assert "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in rendered
    assert "## Source Coverage Plan" in rendered
    assert "### Person-Box Reconciliation" in rendered
    assert "`apron_required` | `candidate_person_mapping_present_pending_review`" in rendered
    assert "`harness_required` | `candidate_person_mapping_present_pending_review`" in rendered


def test_seed_source_fingerprint_ignores_review_evidence_template_metadata(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    before = doctor.source_review_fingerprint(report)

    report["review_evidence_templates"] = {
        "dir": str(tmp_path / "evidence_templates"),
        "files": doctor.write_review_evidence_templates(report, tmp_path / "evidence_templates"),
    }
    report["review_packets"] = {
        "dir": str(tmp_path / "review_packets"),
        "files": doctor.write_review_packets(report, tmp_path / "review_packets"),
    }
    report["import_manifest_review"] = {"ok": False, "gate_passed": False}
    report["next_review_batch_validation"] = {"ok": True, "source_review_sha256_matches": True}
    report["review_bundle_validation"] = {"ok": True, "source_review_sha256_matches": True}

    assert doctor.source_review_fingerprint(report) == before


def test_seed_source_cli_combined_validation_keeps_source_artifact_hashes_stable(
    tmp_path: Path,
    capsys,
):
    doctor = _load_doctor()
    report_path = tmp_path / "seed_source_review.json"
    work_order_path = tmp_path / "seed_source_review.md"
    import_template_path = tmp_path / "seed_import_manifest.template.yaml"
    minimum_import_template_path = tmp_path / "minimum_seed_import_manifest.template.yaml"
    next_batch_path = tmp_path / "next_review_batch.json"
    kickoff_path = tmp_path / "source_review_kickoff.md"
    coverage_path = tmp_path / "source_coverage_plan.json"
    bundle_path = tmp_path / "source_review_bundle.json"
    checklist_path = tmp_path / "seed_source_review_checklist.csv"
    evidence_dir = tmp_path / "evidence"
    packet_dir = tmp_path / "packets"
    generate_exit_code = doctor.main(
        [
            "--out",
            str(report_path),
            "--work-order-out",
            str(work_order_path),
            "--import-template-out",
            str(import_template_path),
            "--minimum-import-template-out",
            str(minimum_import_template_path),
            "--review-checklist-csv-out",
            str(checklist_path),
            "--review-evidence-template-dir",
            str(evidence_dir),
            "--review-packet-dir",
            str(packet_dir),
            "--next-review-batch-out",
            str(next_batch_path),
            "--review-kickoff-out",
            str(kickoff_path),
            "--source-coverage-plan-out",
            str(coverage_path),
            "--review-bundle-out",
            str(bundle_path),
        ]
    )
    capsys.readouterr()
    assert generate_exit_code == 0
    work_order_text = work_order_path.read_text(encoding="utf-8")
    assert "--validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in work_order_text
    assert "IMPORT_MANIFEST: gate=pass" in work_order_text
    assert "IMPORT_MANIFEST: gate=blocked" in work_order_text
    assert "exit nonzero" in work_order_text

    exit_code = doctor.main(
        [
            "--review-bundle-out",
            "",
            "--validate-next-review-batch",
            str(next_batch_path),
            "--validate-review-bundle",
            str(bundle_path),
            "--validate-import-manifest",
            str(import_template_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "IMPORT_MANIFEST: gate=blocked included=0 approved=0" in output
    assert "NEXT_REVIEW_BATCH: ok=True" in output
    assert "REVIEW_BUNDLE: ok=True" in output
    assert "source_sha_match=True" in output


def test_seed_source_cli_import_validation_does_not_write_default_handoffs(
    tmp_path: Path,
    capsys,
):
    doctor = _load_doctor()
    report_path = tmp_path / "generated_seed_source_review.json"
    import_template_path = tmp_path / "generated_seed_import_manifest.template.yaml"
    default_report_path = tmp_path / "default_seed_source_review.json"
    default_work_order_path = tmp_path / "default_seed_source_review.md"
    default_import_template_path = tmp_path / "default_seed_import_manifest.template.yaml"
    default_minimum_import_template_path = tmp_path / "default_minimum_seed_import_manifest.template.yaml"
    default_review_checklist_path = tmp_path / "default_seed_source_review_checklist.csv"
    default_review_evidence_template_dir = tmp_path / "default_seed_source_review_evidence"
    default_review_packet_dir = tmp_path / "default_seed_source_review_packets"
    default_next_review_batch_path = tmp_path / "default_next_source_review_batch.json"
    default_review_kickoff_path = tmp_path / "default_source_review_kickoff.md"
    default_source_coverage_plan_path = tmp_path / "default_source_coverage_plan.json"
    default_review_bundle_path = tmp_path / "default_source_review_bundle.json"
    default_report_path.write_text("sentinel-source-review", encoding="utf-8")
    default_work_order_path.write_text("sentinel-work-order", encoding="utf-8")
    default_import_template_path.write_text("sentinel-import-template", encoding="utf-8")
    default_minimum_import_template_path.write_text("sentinel-minimum-import-template", encoding="utf-8")
    default_review_checklist_path.write_text("sentinel-review-checklist", encoding="utf-8")
    default_review_evidence_template_dir.mkdir()
    (default_review_evidence_template_dir / "sentinel.yaml").write_text(
        "sentinel-review-evidence-template",
        encoding="utf-8",
    )
    default_review_packet_dir.mkdir()
    (default_review_packet_dir / "sentinel.md").write_text("sentinel-review-packet", encoding="utf-8")
    default_next_review_batch_path.write_text("sentinel-next-review-batch", encoding="utf-8")
    default_review_kickoff_path.write_text("sentinel-review-kickoff", encoding="utf-8")
    default_source_coverage_plan_path.write_text("sentinel-source-coverage-plan", encoding="utf-8")
    default_review_bundle_path.write_text("sentinel-review-bundle", encoding="utf-8")

    original_defaults = {
        "DEFAULT_OUT": doctor.DEFAULT_OUT,
        "DEFAULT_WORK_ORDER": doctor.DEFAULT_WORK_ORDER,
        "DEFAULT_IMPORT_TEMPLATE": doctor.DEFAULT_IMPORT_TEMPLATE,
        "DEFAULT_MINIMUM_IMPORT_TEMPLATE": doctor.DEFAULT_MINIMUM_IMPORT_TEMPLATE,
        "DEFAULT_REVIEW_CHECKLIST_CSV": doctor.DEFAULT_REVIEW_CHECKLIST_CSV,
        "DEFAULT_REVIEW_EVIDENCE_TEMPLATE_DIR": doctor.DEFAULT_REVIEW_EVIDENCE_TEMPLATE_DIR,
        "DEFAULT_REVIEW_PACKET_DIR": doctor.DEFAULT_REVIEW_PACKET_DIR,
        "DEFAULT_NEXT_REVIEW_BATCH": doctor.DEFAULT_NEXT_REVIEW_BATCH,
        "DEFAULT_REVIEW_KICKOFF": doctor.DEFAULT_REVIEW_KICKOFF,
        "DEFAULT_SOURCE_COVERAGE_PLAN": doctor.DEFAULT_SOURCE_COVERAGE_PLAN,
        "DEFAULT_REVIEW_BUNDLE": doctor.DEFAULT_REVIEW_BUNDLE,
    }
    doctor.DEFAULT_OUT = default_report_path
    doctor.DEFAULT_WORK_ORDER = default_work_order_path
    doctor.DEFAULT_IMPORT_TEMPLATE = default_import_template_path
    doctor.DEFAULT_MINIMUM_IMPORT_TEMPLATE = default_minimum_import_template_path
    doctor.DEFAULT_REVIEW_CHECKLIST_CSV = default_review_checklist_path
    doctor.DEFAULT_REVIEW_EVIDENCE_TEMPLATE_DIR = default_review_evidence_template_dir
    doctor.DEFAULT_REVIEW_PACKET_DIR = default_review_packet_dir
    doctor.DEFAULT_NEXT_REVIEW_BATCH = default_next_review_batch_path
    doctor.DEFAULT_REVIEW_KICKOFF = default_review_kickoff_path
    doctor.DEFAULT_SOURCE_COVERAGE_PLAN = default_source_coverage_plan_path
    doctor.DEFAULT_REVIEW_BUNDLE = default_review_bundle_path
    try:
        generate_exit_code = doctor.main(
            [
                "--out",
                str(report_path),
                "--import-template-out",
                str(import_template_path),
                "--work-order-out",
                "",
                "--minimum-import-template-out",
                "",
                "--review-checklist-csv-out",
                "",
                "--review-evidence-template-dir",
                "",
                "--review-packet-dir",
                "",
                "--next-review-batch-out",
                "",
                "--review-kickoff-out",
                "",
                "--source-coverage-plan-out",
                "",
                "--review-bundle-out",
                "",
            ]
        )
        capsys.readouterr()
        assert generate_exit_code == 0

        validation_exit_code = doctor.main(
            [
                "--validate-import-manifest",
                str(import_template_path),
            ]
        )
    finally:
        for name, value in original_defaults.items():
            setattr(doctor, name, value)

    output = capsys.readouterr().out
    assert validation_exit_code == 1
    assert "IMPORT_MANIFEST: gate=blocked included=0 approved=0" in output
    assert default_report_path.read_text(encoding="utf-8") == "sentinel-source-review"
    assert default_work_order_path.read_text(encoding="utf-8") == "sentinel-work-order"
    assert default_import_template_path.read_text(encoding="utf-8") == "sentinel-import-template"
    assert (
        default_minimum_import_template_path.read_text(encoding="utf-8")
        == "sentinel-minimum-import-template"
    )
    assert default_review_checklist_path.read_text(encoding="utf-8") == "sentinel-review-checklist"
    assert sorted(path.name for path in default_review_evidence_template_dir.iterdir()) == [
        "sentinel.yaml"
    ]
    assert (
        default_review_evidence_template_dir / "sentinel.yaml"
    ).read_text(encoding="utf-8") == "sentinel-review-evidence-template"
    assert sorted(path.name for path in default_review_packet_dir.iterdir()) == ["sentinel.md"]
    assert (default_review_packet_dir / "sentinel.md").read_text(encoding="utf-8") == (
        "sentinel-review-packet"
    )
    assert default_next_review_batch_path.read_text(encoding="utf-8") == "sentinel-next-review-batch"
    assert default_review_kickoff_path.read_text(encoding="utf-8") == "sentinel-review-kickoff"
    assert (
        default_source_coverage_plan_path.read_text(encoding="utf-8")
        == "sentinel-source-coverage-plan"
    )
    assert default_review_bundle_path.read_text(encoding="utf-8") == "sentinel-review-bundle"


def test_seed_source_doctor_rejects_missing_shared_source(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidates = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"]
    candidates[0]["source_ref"] = "missing_public_source"
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor.audit_seed_sources(path)

    assert report["ok"] is False
    assert report["gate_passed"] is False
    assert any("source_ref does not exist" in error for error in report["errors"])


def test_seed_source_doctor_rejects_stale_source_research(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    first_candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    source = model_packs["shared_sources"][first_candidate["source_ref"]]
    source["checked"] = "2026-06-01"
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor._audit_seed_sources_doc(
        model_packs,
        path,
        now=datetime(2026, 6, 22, tzinfo=timezone.utc),
    )

    assert report["ok"] is False
    assert any("shared source research is stale" in error for error in report["errors"])
    assert any("max_age_days=14" in error for error in report["errors"])


def test_seed_source_doctor_rejects_invalid_checked_date(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    first_candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    source = model_packs["shared_sources"][first_candidate["source_ref"]]
    source["checked"] = "recently"
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor._audit_seed_sources_doc(
        model_packs,
        path,
        now=datetime(2026, 6, 22, tzinfo=timezone.utc),
    )

    assert report["ok"] is False
    assert any("shared source checked must be YYYY-MM-DD" in error for error in report["errors"])


def test_seed_source_doctor_rejects_invalid_research_max_age_policy(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    model_packs["policy"]["source_research_max_age_days"] = 0
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor._audit_seed_sources_doc(
        model_packs,
        path,
        now=datetime(2026, 6, 22, tzinfo=timezone.utc),
    )

    assert report["ok"] is False
    assert any(
        "policy.source_research_max_age_days must be a positive integer" in error
        for error in report["errors"]
    )


def test_seed_source_doctor_requires_full_review_checklist(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["required_review"] = ["license_terms"]
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor.audit_seed_sources(path)

    assert report["ok"] is False
    assert any("required_review missing items" in error for error in report["errors"])


def test_seed_source_doctor_requires_review_priority_and_focus(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate.pop("review_focus", None)
    candidate["review_priority"] = 0
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor.audit_seed_sources(path)

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "review_priority must be a positive integer" in errors
    assert "review_focus is required" in errors


def test_seed_source_doctor_requires_completed_review_for_approved_source(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor.audit_seed_sources(path)

    assert report["ok"] is False
    assert any("completed_review missing approvals" in error for error in report["errors"])


def test_seed_source_doctor_requires_source_research_evidence_for_approved_source(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    source = model_packs["shared_sources"][candidate["source_ref"]]
    source["source_research_evidence"] = []
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor.audit_seed_sources(path)

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "approved training source requires evidence-ready source research" in errors
    assert "source_research_evidence" in errors
    assert report["candidates"][0]["training_usable"] is False


def test_seed_source_import_manifest_blocks_unapproved_source(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    import_manifest = doctor.build_import_manifest_template(report)
    import_manifest["imports"][0]["include_in_training"] = True
    path = tmp_path / "seed_import.yaml"
    path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(path, report)

    assert import_review["ok"] is False
    assert import_review["gate_passed"] is False
    errors = "\n".join(import_review["errors"])
    blockers = "\n".join(import_review["blockers"])
    assert "review_status must be approved_for_training" in errors
    assert "completed_review missing approvals" in errors
    assert "seed source review is not training-usable" in blockers


def test_seed_source_review_checklist_apply_updates_model_packs_with_completed_review(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    rows = list(csv.DictReader(doctor.build_review_checklist_csv(report).splitlines()))
    rows[0]["approval_status"] = "approved_for_training"
    rows[0]["approved_for_training"] = "true"
    rows[0]["training_usable"] = "true"
    rows[0]["current_blocker"] = ""
    rows[0]["reviewed_by"] = "qa_reviewer"
    rows[0]["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    rows[0]["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    _attach_row_review_evidence(rows[0], tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    rows[0]["review_notes"] = "Legal/export/provenance review completed for this seed source."
    for field in doctor.REVIEW_BOOLEAN_FIELDS:
        rows[0][field] = "true"
    checklist_path = tmp_path / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.REVIEW_CHECKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    apply_report = doctor.apply_review_checklist_to_model_packs(
        model_packs_path=MODEL_PACKS_PATH,
        checklist_csv_path=checklist_path,
    )

    assert apply_report["ok"] is True
    assert apply_report["row_count"] == len(rows)
    assert apply_report["applied_count"] == len(rows)
    assert apply_report["approved_count"] == 1
    updated = apply_report["updated_model_packs"]
    candidate = next(
        item
        for item in updated["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"]
        if item["source_ref"] == rows[0]["source_ref"] and item["capability"] == rows[0]["capability"]
    )
    assert candidate["approval_status"] == "approved_for_training"
    assert candidate["approved_for_training"] is True
    assert candidate["blocker"] == ""
    assert candidate["completed_review"] == {field: True for field in doctor.REVIEW_BOOLEAN_FIELDS}
    assert candidate["review_evidence_path"] == rows[0]["review_evidence_path"]
    assert candidate["review_evidence_sha256"] == rows[0]["review_evidence_sha256"]
    assert len(candidate["review_evidence_sha256"]) == 64
    assert candidate["review_notes"] == "Legal/export/provenance review completed for this seed source."
    assert apply_report["updated_audit"]["ok"] is True
    assert apply_report["updated_audit"]["gate_passed"] is False


def test_seed_source_review_checklist_apply_rejects_incomplete_approved_review(tmp_path: Path):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    rows = list(csv.DictReader(doctor.build_review_checklist_csv(report).splitlines()))
    rows[0]["approval_status"] = "approved_for_training"
    rows[0]["approved_for_training"] = "true"
    rows[0]["training_usable"] = "true"
    rows[0]["current_blocker"] = ""
    rows[0]["reviewed_by"] = "qa_reviewer"
    rows[0]["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    rows[0]["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    _attach_row_review_evidence(rows[0], tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    for field in doctor.REVIEW_BOOLEAN_FIELDS:
        rows[0][field] = "true"
    rows[0]["export_terms"] = "false"
    checklist_path = tmp_path / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.REVIEW_CHECKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    apply_report = doctor.apply_review_checklist_to_model_packs(
        model_packs_path=MODEL_PACKS_PATH,
        checklist_csv_path=checklist_path,
    )

    assert apply_report["ok"] is False
    assert apply_report["updated_model_packs"] is None
    assert any("completed review approvals missing: export_terms" in error for error in apply_report["errors"])


def test_seed_source_review_checklist_apply_rejects_approved_review_without_evidence(
    tmp_path: Path,
):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    rows = list(csv.DictReader(doctor.build_review_checklist_csv(report).splitlines()))
    rows[0]["approval_status"] = "approved_for_training"
    rows[0]["approved_for_training"] = "true"
    rows[0]["training_usable"] = "true"
    rows[0]["current_blocker"] = ""
    rows[0]["reviewed_by"] = "qa_reviewer"
    rows[0]["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    rows[0]["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    for field in doctor.REVIEW_BOOLEAN_FIELDS:
        rows[0][field] = "true"
    checklist_path = tmp_path / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.REVIEW_CHECKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    apply_report = doctor.apply_review_checklist_to_model_packs(
        model_packs_path=MODEL_PACKS_PATH,
        checklist_csv_path=checklist_path,
    )

    assert apply_report["ok"] is False
    assert apply_report["updated_model_packs"] is None
    errors = "\n".join(apply_report["errors"])
    assert "review_evidence_path is required for approved sources" in errors
    assert "review_evidence_sha256 is required for approved sources" in errors


def test_seed_source_review_checklist_apply_rejects_malformed_evidence_bundle(
    tmp_path: Path,
):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    rows = list(csv.DictReader(doctor.build_review_checklist_csv(report).splitlines()))
    rows[0]["approval_status"] = "approved_for_training"
    rows[0]["approved_for_training"] = "true"
    rows[0]["training_usable"] = "true"
    rows[0]["current_blocker"] = ""
    rows[0]["reviewed_by"] = "qa_reviewer"
    rows[0]["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    rows[0]["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    evidence_path = tmp_path / "malformed_review_evidence.yaml"
    evidence_path.write_text(
        yaml.safe_dump(
            {
                "kind": "apron_harness_seed_source_review_evidence",
                "version": 1,
                "source_ref": rows[0]["source_ref"],
                "capability": rows[0]["capability"],
                "reviewed_by": "qa_reviewer",
                "reviewed_at": "2026-06-22T00:00:00+00:00",
                "review_items": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows[0]["review_evidence_path"] = str(evidence_path)
    rows[0]["review_evidence_sha256"] = _sha256_file(evidence_path)
    for field in doctor.REVIEW_BOOLEAN_FIELDS:
        rows[0][field] = "true"
    checklist_path = tmp_path / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.REVIEW_CHECKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    apply_report = doctor.apply_review_checklist_to_model_packs(
        model_packs_path=MODEL_PACKS_PATH,
        checklist_csv_path=checklist_path,
    )

    assert apply_report["ok"] is False
    assert apply_report["updated_model_packs"] is None
    errors = "\n".join(apply_report["errors"])
    assert "review evidence review_items are required for approved sources" in errors
    assert "review evidence license_terms.approved must be true for approved sources" in errors


def test_seed_source_review_checklist_apply_rejects_approved_agent_prefill(
    tmp_path: Path,
):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    rows = list(csv.DictReader(doctor.build_review_checklist_csv(report).splitlines()))
    rows[0]["approval_status"] = "approved_for_training"
    rows[0]["approved_for_training"] = "true"
    rows[0]["training_usable"] = "true"
    rows[0]["current_blocker"] = ""
    rows[0]["reviewed_by"] = "qa_reviewer"
    rows[0]["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    rows[0]["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    evidence_doc = _review_evidence_payload(
        review_fields=doctor.REVIEW_BOOLEAN_FIELDS,
        source_ref=rows[0]["source_ref"],
        capability=rows[0]["capability"],
        source_url=rows[0]["source_url"],
        license_note=rows[0]["license_note"],
        reviewed_by="qa_reviewer",
        reviewed_at="2026-06-22T00:00:00+00:00",
    )
    evidence_doc["review_items"]["license_terms"]["prefilled_by_agent"] = True
    evidence_path = tmp_path / "agent_prefilled_review_evidence.yaml"
    evidence_path.write_text(yaml.safe_dump(evidence_doc, sort_keys=False), encoding="utf-8")
    rows[0]["review_evidence_path"] = str(evidence_path)
    rows[0]["review_evidence_sha256"] = _sha256_file(evidence_path)
    for field in doctor.REVIEW_BOOLEAN_FIELDS:
        rows[0][field] = "true"
    checklist_path = tmp_path / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.REVIEW_CHECKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    apply_report = doctor.apply_review_checklist_to_model_packs(
        model_packs_path=MODEL_PACKS_PATH,
        checklist_csv_path=checklist_path,
    )

    assert apply_report["ok"] is False
    assert apply_report["updated_model_packs"] is None
    errors = "\n".join(apply_report["errors"])
    assert "review evidence license_terms.prefilled_by_agent must be false or removed" in errors


def test_seed_source_review_checklist_apply_rejects_evidence_source_context_mismatch(
    tmp_path: Path,
):
    doctor = _load_doctor()
    report = doctor.audit_seed_sources(MODEL_PACKS_PATH)
    rows = list(csv.DictReader(doctor.build_review_checklist_csv(report).splitlines()))
    rows[0]["approval_status"] = "approved_for_training"
    rows[0]["approved_for_training"] = "true"
    rows[0]["training_usable"] = "true"
    rows[0]["current_blocker"] = ""
    rows[0]["reviewed_by"] = "qa_reviewer"
    rows[0]["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    rows[0]["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    evidence_doc = _review_evidence_payload(
        review_fields=doctor.REVIEW_BOOLEAN_FIELDS,
        source_ref=rows[0]["source_ref"],
        capability=rows[0]["capability"],
        source_url="https://example.com/wrong-source",
        license_note="Different license note",
        reviewed_by="qa_reviewer",
        reviewed_at="2026-06-22T00:00:00+00:00",
    )
    evidence_path = tmp_path / "wrong_source_context_review_evidence.yaml"
    evidence_path.write_text(yaml.safe_dump(evidence_doc, sort_keys=False), encoding="utf-8")
    rows[0]["review_evidence_path"] = str(evidence_path)
    rows[0]["review_evidence_sha256"] = _sha256_file(evidence_path)
    for field in doctor.REVIEW_BOOLEAN_FIELDS:
        rows[0][field] = "true"
    checklist_path = tmp_path / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.REVIEW_CHECKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    apply_report = doctor.apply_review_checklist_to_model_packs(
        model_packs_path=MODEL_PACKS_PATH,
        checklist_csv_path=checklist_path,
    )

    assert apply_report["ok"] is False
    assert apply_report["updated_model_packs"] is None
    errors = "\n".join(apply_report["errors"])
    assert "review evidence source_url must match the approved source" in errors
    assert "review evidence license_note must match the approved source" in errors


def test_seed_source_import_manifest_accepts_approved_training_source(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")
    report = doctor.audit_seed_sources(path)
    source = report["candidates"][0]
    export_zip = _write_yolo_export_zip(
        tmp_path / "harness_seed_yolo_export.zip",
        names=["top", "safety-harness", "lanyard"],
        label_classes=["top", "safety-harness", "lanyard"],
    )
    export_sha256 = _sha256_file(export_zip)
    import_manifest = {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": doctor.source_review_fingerprint(report),
        "imports": [
            {
                "source_ref": source["source_ref"],
                "capability": source["capability"],
                "include_in_training": True,
                "review_status": "approved_for_training",
                "reviewed_by": "qa_reviewer",
                "reviewed_at": "2026-06-22T00:00:00+00:00",
                "manifest_import_path": source["manifest_import_path"],
                "raw_export_ref": "s3://cleared-seed-exports/harness_seed_yolo_export.zip",
                "raw_export_sha256": export_sha256,
                "raw_export_local_path": str(export_zip),
                "export_format": "yolo",
                "completed_review": {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS},
                "class_mapping": {
                    "top": "person",
                    "safety-harness": "safety_harness",
                    "lanyard": "safety_lanyard",
                },
                "person_box_policy": "Source top boxes were reviewed and mapped to person boxes.",
                "hard_negative_policy": "Hard negatives are tracked in the capture manifest for non-harness straps.",
                "split_plan": {
                    "train": "source-defined train split reconciled to manifest",
                    "val": "source-defined val split reconciled to manifest",
                    "test": "held-out test split reconciled to manifest",
                },
                "expected_labeled_images_per_class": {
                    "person": 3,
                    "apron": 0,
                    "safety_harness": 3,
                    "safety_lanyard": 3,
                },
            }
        ],
    }
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is True
    assert import_review["gate_passed"] is True
    assert import_review["included_count"] == 1
    assert import_review["approved_count"] == 1
    assert import_review["source_review_sha256"] == doctor.source_review_fingerprint(report)
    assert import_review["source_review_sha256_matches"] is True
    assert import_review["imports"][0]["raw_export_ref"] == (
        "s3://cleared-seed-exports/harness_seed_yolo_export.zip"
    )
    assert import_review["imports"][0]["raw_export_sha256"] == export_sha256
    assert import_review["imports"][0]["yolo_export_preflight"]["sha256"] == export_sha256
    assert import_review["imports"][0]["yolo_export_preflight"]["label_file_count_by_local_class"] == {
        "person": 3,
        "safety_harness": 3,
        "safety_lanyard": 3,
    }
    assert import_review["errors"] == []
    assert import_review["blockers"] == []


def test_seed_source_import_manifest_validates_generated_review_artifacts(tmp_path: Path):
    doctor = _load_doctor()
    report, source = _approved_harness_seed_report(doctor, tmp_path)
    review_artifacts = _add_generated_review_artifacts(
        doctor,
        report,
        tmp_path,
        source_ref=source["source_ref"],
        capability=source["capability"],
    )
    export_zip = _write_yolo_export_zip(
        tmp_path / "harness_seed_yolo_export.zip",
        names=["top", "safety-harness", "lanyard"],
        label_classes=["top", "safety-harness", "lanyard"],
    )
    export_sha256 = _sha256_file(export_zip)
    import_manifest = _valid_harness_seed_import(
        source,
        source_review_sha256=doctor.source_review_fingerprint(report),
        export_zip=export_zip,
        export_sha256=export_sha256,
        review_artifacts=review_artifacts,
    )
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is True
    assert import_review["gate_passed"] is True
    assert import_review["imports"][0]["review_artifact_preflight"]["checked"] is True
    assert import_review["imports"][0]["review_artifact_preflight"]["errors"] == []


def test_seed_source_import_manifest_rejects_yolo_export_orphan_labels(tmp_path: Path):
    doctor = _load_doctor()
    report, source = _approved_harness_seed_report(doctor, tmp_path)
    export_zip = _write_orphan_label_yolo_export_zip(
        tmp_path / "harness_seed_yolo_export.zip",
        names=["top", "safety-harness", "lanyard"],
        label_classes=["top", "safety-harness", "lanyard"],
    )
    export_sha256 = _sha256_file(export_zip)
    import_manifest = _valid_harness_seed_import(
        source,
        source_review_sha256=doctor.source_review_fingerprint(report),
        export_zip=export_zip,
        export_sha256=export_sha256,
    )
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is False
    errors = "\n".join(import_review["errors"])
    assert "YOLO export train labels must have matching images: train_orphan_label" in errors
    assert "YOLO export valid labels must have matching images: valid_orphan_label" in errors
    assert "YOLO export test labels must have matching images: test_orphan_label" in errors
    assert import_review["imports"][0]["yolo_export_preflight"]["orphan_label_count_by_split"] == {
        "test": 1,
        "train": 1,
        "valid": 1,
    }


def test_seed_source_import_manifest_rejects_reviewer_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    report, source = _approved_harness_seed_report(doctor, tmp_path)
    review_artifacts = _add_generated_review_artifacts(
        doctor,
        report,
        tmp_path,
        source_ref=source["source_ref"],
        capability=source["capability"],
    )
    export_zip = _write_yolo_export_zip(
        tmp_path / "harness_seed_yolo_export.zip",
        names=["top", "safety-harness", "lanyard"],
        label_classes=["top", "safety-harness", "lanyard"],
    )
    export_sha256 = _sha256_file(export_zip)
    import_manifest = _valid_harness_seed_import(
        source,
        source_review_sha256=doctor.source_review_fingerprint(report),
        export_zip=export_zip,
        export_sha256=export_sha256,
        review_artifacts=review_artifacts,
    )
    import_manifest["imports"][0]["reviewed_by"] = "different_reviewer"
    import_manifest["imports"][0]["reviewed_at"] = "2026-06-23T00:00:00+00:00"
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is False
    errors = "\n".join(import_review["errors"])
    assert "reviewed_by must match the approved seed source review" in errors
    assert "reviewed_at must match the approved seed source review" in errors


def test_seed_source_import_manifest_rejects_stale_generated_review_artifacts(tmp_path: Path):
    doctor = _load_doctor()
    report, source = _approved_harness_seed_report(doctor, tmp_path)
    review_artifacts = _add_generated_review_artifacts(
        doctor,
        report,
        tmp_path,
        source_ref=source["source_ref"],
        capability=source["capability"],
    )
    review_artifacts["review_packet_sha256"] = "0" * 64
    export_zip = _write_yolo_export_zip(
        tmp_path / "harness_seed_yolo_export.zip",
        names=["top", "safety-harness", "lanyard"],
        label_classes=["top", "safety-harness", "lanyard"],
    )
    import_manifest = _valid_harness_seed_import(
        source,
        source_review_sha256=doctor.source_review_fingerprint(report),
        export_zip=export_zip,
        export_sha256=_sha256_file(export_zip),
        review_artifacts=review_artifacts,
    )
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is False
    assert any(
        "review_artifacts.review_packet_sha256 must match the seed source review artifact SHA-256"
        in error
        for error in import_review["errors"]
    )
    assert import_review["imports"][0]["review_artifact_preflight"]["checked"] is True


def test_seed_source_import_manifest_rejects_mismatched_source_review_sha256(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")
    report = doctor.audit_seed_sources(path)
    source = report["candidates"][0]
    import_manifest = {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": "0" * 64,
        "imports": [
            {
                "source_ref": source["source_ref"],
                "capability": source["capability"],
                "include_in_training": True,
                "review_status": "approved_for_training",
                "reviewed_by": "qa_reviewer",
                "reviewed_at": "2026-06-22T00:00:00+00:00",
                "manifest_import_path": source["manifest_import_path"],
                "raw_export_ref": "s3://cleared-seed-exports/harness_seed_yolo_export.zip",
                "raw_export_sha256": "a" * 64,
                "export_format": "yolo",
                "completed_review": {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS},
                "class_mapping": {
                    "top": "person",
                    "safety-harness": "safety_harness",
                    "lanyard": "safety_lanyard",
                },
                "person_box_policy": "Source top boxes were reviewed and mapped to person boxes.",
                "hard_negative_policy": "Hard negatives are tracked in the capture manifest.",
                "split_plan": {
                    "train": "source-defined train split reconciled to manifest",
                    "val": "source-defined val split reconciled to manifest",
                    "test": "held-out test split reconciled to manifest",
                },
                "expected_labeled_images_per_class": {
                    "person": 65,
                    "apron": 0,
                    "safety_harness": 65,
                    "safety_lanyard": 12,
                },
            }
        ],
    }
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is False
    assert any("source_review_sha256 does not match seed source review" in error for error in import_review["errors"])


def test_seed_source_import_manifest_rejects_local_raw_export_ref(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")
    report = doctor.audit_seed_sources(path)
    source = report["candidates"][0]
    import_manifest = {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": doctor.source_review_fingerprint(report),
        "imports": [
            {
                "source_ref": source["source_ref"],
                "capability": source["capability"],
                "include_in_training": True,
                "review_status": "approved_for_training",
                "reviewed_by": "qa_reviewer",
                "reviewed_at": "2026-06-22T00:00:00+00:00",
                "manifest_import_path": source["manifest_import_path"],
                "raw_export_ref": "qa/video_eval/datasets/imported/harness_seed_yolo_export.zip",
                "raw_export_sha256": "a" * 64,
                "export_format": "yolo",
                "completed_review": {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS},
                "class_mapping": {
                    "top": "person",
                    "safety-harness": "safety_harness",
                    "lanyard": "safety_lanyard",
                },
                "person_box_policy": "Source top boxes were reviewed and mapped to person boxes.",
                "hard_negative_policy": "Hard negatives are tracked in the capture manifest.",
                "split_plan": {
                    "train": "source-defined train split reconciled to manifest",
                    "val": "source-defined val split reconciled to manifest",
                    "test": "held-out test split reconciled to manifest",
                },
                "expected_labeled_images_per_class": {
                    "person": 65,
                    "apron": 0,
                    "safety_harness": 65,
                    "safety_lanyard": 12,
                },
            }
        ],
    }
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is False
    assert any(
        "raw_export_ref must be a remote immutable export reference" in error
        for error in import_review["errors"]
    )


def test_seed_source_import_manifest_requires_export_policy_and_expected_counts(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")
    report = doctor.audit_seed_sources(path)
    source = report["candidates"][0]
    import_manifest = {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": doctor.source_review_fingerprint(report),
        "imports": [
            {
                "source_ref": source["source_ref"],
                "capability": source["capability"],
                "include_in_training": True,
                "review_status": "approved_for_training",
                "reviewed_by": "qa_reviewer",
                "reviewed_at": "2026-06-22T00:00:00+00:00",
                "manifest_import_path": source["manifest_import_path"],
                "completed_review": {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS},
                "class_mapping": {"safety-harness": "safety_harness"},
                "split_plan": {
                    "train": "source-defined train split reconciled to manifest",
                    "val": "source-defined val split reconciled to manifest",
                    "test": "held-out test split reconciled to manifest",
                },
            }
        ],
    }
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is False
    errors = "\n".join(import_review["errors"])
    assert "raw_export_ref is required" in errors
    assert "raw_export_sha256 is required" in errors
    assert "raw_export_local_path is required" in errors
    assert "person_box_policy is required" in errors
    assert "hard_negative_policy is required" in errors
    assert "expected_labeled_images_per_class.person must be greater than 0" in errors
    assert "expected_labeled_images_per_class.safety_lanyard must be greater than 0" in errors


def test_seed_source_import_manifest_requires_class_mapping_to_cover_required_local_classes(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    candidate["blocker"] = ""
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")
    report = doctor.audit_seed_sources(path)
    source = report["candidates"][0]
    import_manifest = {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": doctor.source_review_fingerprint(report),
        "imports": [
            {
                "source_ref": source["source_ref"],
                "capability": source["capability"],
                "include_in_training": True,
                "review_status": "approved_for_training",
                "reviewed_by": "qa_reviewer",
                "reviewed_at": "2026-06-22T00:00:00+00:00",
                "manifest_import_path": source["manifest_import_path"],
                "raw_export_ref": "s3://cleared-seed-exports/harness_seed_yolo_export.zip",
                "raw_export_sha256": "a" * 64,
                "export_format": "yolo",
                "completed_review": {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS},
                "class_mapping": {"safety-harness": "safety_harness"},
                "person_box_policy": "Source person boxes were reviewed.",
                "hard_negative_policy": "Hard negatives are tracked in the capture manifest.",
                "split_plan": {
                    "train": "source-defined train split reconciled to manifest",
                    "val": "source-defined val split reconciled to manifest",
                    "test": "held-out test split reconciled to manifest",
                },
                "expected_labeled_images_per_class": {
                    "person": 65,
                    "apron": 0,
                    "safety_harness": 65,
                    "safety_lanyard": 65,
                },
            }
        ],
    }
    import_path = tmp_path / "seed_import.yaml"
    import_path.write_text(yaml.safe_dump(import_manifest, sort_keys=False), encoding="utf-8")

    import_review = doctor.validate_import_manifest(import_path, report)

    assert import_review["ok"] is False
    assert any(
        "class_mapping must map source classes to required local classes: person" in error
        for error in import_review["errors"]
    )
    assert any(
        "safety_lanyard" in error
        for error in import_review["errors"]
    )


def test_seed_source_doctor_rejects_approved_source_that_keeps_blocker(tmp_path: Path):
    doctor = _load_doctor()
    model_packs = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    candidate = model_packs["packs"]["factory_ppe_3cam"]["sourcing_status"]["candidate_sources"][0]
    candidate["approval_status"] = "approved_for_training"
    candidate["approved_for_training"] = True
    candidate["manifest_import_path"] = "qa/video_eval/datasets/imported/harness_seed_manifest.yaml"
    candidate["reviewed_by"] = "qa_reviewer"
    candidate["reviewed_at"] = "2026-06-22T00:00:00+00:00"
    _attach_candidate_review_evidence(candidate, tmp_path, doctor.REVIEW_BOOLEAN_FIELDS)
    candidate["completed_review"] = {item: True for item in doctor.REVIEW_BOOLEAN_FIELDS}
    path = tmp_path / "model_packs.yaml"
    path.write_text(yaml.safe_dump(model_packs, sort_keys=False), encoding="utf-8")

    report = doctor.audit_seed_sources(path)

    assert report["ok"] is False
    assert any("approved training source must clear blocker" in error for error in report["errors"])
