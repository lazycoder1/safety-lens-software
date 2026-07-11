#!/usr/bin/env python3
"""Audit public apron/harness seed sources before dataset import."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PACKS = ROOT / "qa" / "video_eval" / "model_packs.yaml"
DEFAULT_OUT = ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_source_review.json"
DEFAULT_WORK_ORDER = ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_source_review.md"
DEFAULT_IMPORT_TEMPLATE = ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_seed_import_manifest.template.yaml"
DEFAULT_MINIMUM_IMPORT_TEMPLATE = (
    ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_minimum_seed_import_manifest.template.yaml"
)
DEFAULT_REVIEW_CHECKLIST_CSV = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_source_review_checklist.csv"
)
DEFAULT_REVIEW_EVIDENCE_TEMPLATE_DIR = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_source_review_evidence"
)
DEFAULT_REVIEW_PACKET_DIR = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_source_review_packets"
)
DEFAULT_NEXT_REVIEW_BATCH = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_next_source_review_batch.json"
)
DEFAULT_REVIEW_KICKOFF = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_source_review_kickoff.md"
)
DEFAULT_REVIEW_PREFILL_DIR = ROOT / "qa" / "video_eval" / "results"
DEFAULT_REVIEW_BUNDLE = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_source_review_bundle.json"
)
DEFAULT_SOURCE_COVERAGE_PLAN = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_source_coverage_plan.json"
)
REVIEW_EVIDENCE_KIND = "apron_harness_seed_source_review_evidence"
NEXT_REVIEW_BATCH_KIND = "apron_harness_next_source_review_batch"
REVIEW_KICKOFF_KIND = "apron_harness_source_review_kickoff"
REVIEW_BUNDLE_KIND = "apron_harness_seed_source_review_bundle"
SOURCE_COVERAGE_PLAN_KIND = "apron_harness_source_coverage_plan"
DEFAULT_NEXT_REVIEW_BATCH_LIMIT = 5

ALLOWED_CAPABILITIES = {"apron_required", "harness_required"}
HEX_DIGITS = set("0123456789abcdefABCDEF")
ALLOWED_APPROVAL_STATUS = {
    "unreviewed",
    "rejected",
    "approved_for_manifest_import",
    "approved_for_training",
}
REQUIRED_SHARED_SOURCE_FIELDS = {
    "url",
    "checked",
    "license_note",
    "relevance",
    "decision",
}
SOURCE_FACT_REVIEW_PRIORITY_MAX = 50
REQUIRED_SOURCE_FACT_FIELDS = {"classes", "page_kind", "source_author", "task"}
SOURCE_FACT_COUNT_FIELDS = {"image_count", "model_dataset_image_count", "public_page_image_count"}
SOURCE_RESEARCH_EVIDENCE_READY_MIN_ITEMS = 5
ALLOWED_SOURCE_RESEARCH_DISPOSITIONS = {
    "open",
    "unavailable_after_agent_search",
}
REQUIRED_CANDIDATE_FIELDS = {
    "source_ref",
    "capability",
    "status",
    "review_priority",
    "review_focus",
    "blocker",
    "approval_status",
    "approved_for_training",
    "required_review",
}
REQUIRED_REVIEW_ITEMS = {
    "license_terms",
    "export_terms",
    "dataset_card_provenance",
    "privacy_and_identity_risk",
    "class_mapping",
    "person_box_coverage",
    "hard_negative_coverage",
    "train_val_test_split",
    "manifest_import_plan",
}
REVIEW_BOOLEAN_FIELDS = {
    "license_terms",
    "export_terms",
    "dataset_card_provenance",
    "privacy_and_identity_risk",
    "class_mapping",
    "person_box_coverage",
    "hard_negative_coverage",
    "train_val_test_split",
    "manifest_import_plan",
}
BOOL_TRUE_VALUES = {"1", "true", "yes", "y"}
BOOL_FALSE_VALUES = {"", "0", "false", "no", "n"}
REVIEW_CHECKLIST_FIELDS = [
    "review_priority",
    "source_ref",
    "capability",
    "source_url",
    "license_note",
    "approval_status",
    "approved_for_training",
    "training_usable",
    "current_blocker",
    "review_focus",
    "reviewed_by",
    "reviewed_at",
    "manifest_import_path",
    "review_packet_path",
    "review_packet_sha256",
    "review_evidence_template_path",
    "review_evidence_template_sha256",
    "seed_import_manifest_template_path",
    "seed_import_manifest_template_sha256",
    "review_evidence_path",
    "review_evidence_sha256",
    "license_terms",
    "export_terms",
    "dataset_card_provenance",
    "privacy_and_identity_risk",
    "class_mapping",
    "person_box_coverage",
    "hard_negative_coverage",
    "train_val_test_split",
    "manifest_import_plan",
    "review_notes",
]
ALLOWED_IMPORT_EXPORT_FORMATS = {"yolo"}
ALLOWED_RAW_EXPORT_REF_SCHEMES = {"az", "gs", "hf", "https", "oci", "roboflow", "s3"}
REVIEW_EVIDENCE_FIELDS = {"review_evidence_path", "review_evidence_sha256"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_IMPORT_COUNT_CLASSES = {
    "apron_required": {"person", "apron"},
    "harness_required": {"person", "safety_harness", "safety_lanyard"},
}
REVIEW_ITEM_GUIDANCE = {
    "class_mapping": "Map source labels into person plus apron or safety_harness/safety_lanyard, with ignored ambiguous classes documented.",
    "dataset_card_provenance": "Archive source page, dataset card, author, dates, version/export ID, and any Roboflow version metadata used.",
    "export_terms": "Record whether a YOLO export is allowed for commercial local training and where the immutable export artifact is stored.",
    "hard_negative_coverage": "List negative classes and scenarios such as vest-vs-apron, backpack/tool-belt-vs-harness, ropes, cables, and non-PPE people.",
    "license_terms": "Record license, attribution obligations, commercial-use status, and any terms-of-use limits that could affect customer delivery.",
    "manifest_import_plan": "Define manifest_import_path, raw_export_ref, raw_export_sha256, split plan, expected counts, and reviewer ownership.",
    "person_box_coverage": "Verify every training frame has usable person boxes or a documented policy to add/review missing person boxes.",
    "privacy_and_identity_risk": "Check identifiable people, sensitive locations, minors, customer-private footage, and whether any redaction/consent is required.",
    "train_val_test_split": "Document split method, leakage prevention, source-version boundaries, and target minimums for train/val/test.",
}
AGENT_RESEARCH_TASKS = [
    "Open the source page and archive the visible license, author, dataset/version ID, class list, and image counts.",
    "Open the platform terms, Universe license/download docs, export docs, and any linked dataset card or model page.",
    "Record exact URLs, access date, screenshots or saved pages, and contradictions between page license, platform terms, and export workflow.",
    "Check whether YOLO export is available without paid, gated, academic-only, customer-private, or unclear terms.",
    "Inspect the exported class list and sample labels only after the source is approved for review access; do not mark training approval yourself.",
]
HUMAN_APPROVAL_TASKS = [
    "Confirm commercial training rights, attribution obligations, privacy/identity risk, and export permissions.",
    "Confirm class mapping into person/apron/safety_harness/safety_lanyard and document ignored or hard-negative source classes.",
    "Confirm person-box coverage, split leakage controls, and whether the source is appropriate for production or only supplemental seed use.",
    "Sign off the filled review-evidence YAML, checklist row, and seed-import manifest before include_in_training=true.",
]
ROBOFLOW_PLATFORM_REFERENCES = {
    "platform_terms_url": "https://roboflow.com/terms",
    "universe_docs_url": "https://docs.roboflow.com/universe/what-is-roboflow-universe",
    "universe_license_docs_url": "https://docs.roboflow.com/universe/find-a-dataset-on-universe",
    "universe_download_docs_url": "https://docs.roboflow.com/universe/download-a-universe-dataset",
    "export_docs_url": "https://docs.roboflow.com/datasets/dataset-versions/exporting-data",
    "platform_licensing_url": "https://roboflow.com/licensing",
}
LICENSE_REFERENCE_URLS = {
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "Public Domain": "https://creativecommons.org/publicdomain/mark/1.0/",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def source_review_fingerprint(report: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "generated_at",
            "import_manifest_review",
            "next_review_batch",
            "next_review_batch_validation",
            "review_kickoff",
            "review_checklist_csv",
            "review_checklist_apply",
            "review_evidence_templates",
            "review_packets",
            "review_queue",
            "review_bundle",
            "review_bundle_validation",
            "source_coverage_plan",
            "seed_import_manifest_template",
            "minimum_seed_import_manifest_template",
            "work_order",
        }
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _as_string_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if value is None:
        return set()
    return {str(value)}


def _count_value(counts: dict[str, Any], class_name: str) -> int:
    try:
        return int(counts.get(class_name) or 0)
    except (TypeError, ValueError):
        return 0


def _normalized_mapping_values(class_mapping: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for value in class_mapping.values():
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized:
            values.add(normalized)
    return values


def _normalize_review_focus(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in BOOL_TRUE_VALUES:
        return True
    if normalized in BOOL_FALSE_VALUES:
        return False
    return None


def _review_priority(value: Any) -> int | None:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return None
    return priority if priority > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _parse_checked_date(value: Any) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, str]:
    return (
        _review_priority(candidate.get("review_priority")) or 999999,
        str(candidate.get("source_ref") or ""),
    )


def _review_prefill_artifact(candidate: dict[str, Any]) -> dict[str, str]:
    """Return the stable source-review prefill memo artifact when one exists."""
    source_ref = str(candidate.get("source_ref") or "").strip()
    checked_date = _parse_checked_date(candidate.get("checked"))
    if not source_ref or checked_date is None:
        return {"path": "", "sha256": ""}
    source_slug = source_ref.removeprefix("roboflow_")
    date_slug = checked_date.isoformat().replace("-", "_")
    path = DEFAULT_REVIEW_PREFILL_DIR / f"apron_harness_{source_slug}_review_prefill_{date_slug}.md"
    if not path.exists():
        matches = sorted(
            DEFAULT_REVIEW_PREFILL_DIR.glob(f"apron_harness_{source_slug}_review_prefill_*.md"),
            reverse=True,
        )
        path = matches[0] if matches else path
    if not path.exists():
        return {"path": "", "sha256": ""}
    return {
        "path": _rel(path),
        "sha256": _sha256_file(path),
    }


def _source_count_summary(source_facts: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field in sorted(SOURCE_FACT_COUNT_FIELDS):
        count = _positive_int(source_facts.get(field))
        if count is not None:
            counts[field] = count
    return counts


def _source_class_summary(source_facts: dict[str, Any]) -> list[str]:
    raw_classes = source_facts.get("classes") or source_facts.get("relevant_classes") or []
    if not isinstance(raw_classes, list):
        return []
    return [str(item) for item in raw_classes if str(item).strip()]


def _normalize_label(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    prefix, sep, suffix = normalized.partition("_")
    if sep and prefix.isdigit() and suffix:
        return suffix
    return normalized


def _source_label_candidates(source_facts: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for field in ("classes", "relevant_classes"):
        raw_labels = source_facts.get(field)
        if not isinstance(raw_labels, list):
            continue
        for raw in raw_labels:
            label = str(raw or "").strip()
            if label and label not in labels:
                labels.append(label)
    return labels


def _suggested_source_mapping(capability: str, source_facts: dict[str, Any]) -> dict[str, Any]:
    """Return non-approving label-mapping hints for reviewer packets/templates."""
    labels = _source_label_candidates(source_facts)
    suggested: dict[str, list[str]] = {
        "person": [],
        "apron": [],
        "safety_harness": [],
        "safety_lanyard": [],
    }
    hard_negatives: list[str] = []
    ignore_labels: list[str] = []
    warnings: list[str] = []

    for label in labels:
        normalized = _normalize_label(label)
        if normalized in {"person", "human", "body", "worker"}:
            suggested["person"].append(label)
        elif "no_apron" in normalized or normalized in {"without_apron", "noapron"}:
            hard_negatives.append(label)
        elif "apron" in normalized:
            suggested["apron"].append(label)
        elif any(
            token in normalized
            for token in (
                "no_belt",
                "no_front_seftybelt",
                "no_harness",
                "no_safety_belt",
                "no_safety_harness",
                "noharness",
            )
        ):
            hard_negatives.append(label)
        elif "lanyard" in normalized or "lifeline" in normalized or "shock_basorber" in normalized:
            suggested["safety_lanyard"].append(label)
        elif "harness" in normalized or "safety_belt" in normalized or "seftybelt" in normalized:
            suggested["safety_harness"].append(label)
        elif normalized in {
            "backpack",
            "belt",
            "cable",
            "cockroach",
            "gloves",
            "hairnet",
            "helmet",
            "ladder",
            "lizard",
            "mask",
            "mewp",
            "rat",
            "rope",
            "safety_vest",
            "scaffolding",
            "vest",
        } or normalized.startswith("no_"):
            hard_negatives.append(label)
        else:
            ignore_labels.append(label)

    required_classes = REQUIRED_IMPORT_COUNT_CLASSES.get(capability, set())
    missing_required = sorted(
        local_class for local_class in required_classes if not suggested.get(local_class)
    )
    if missing_required:
        warnings.append(
            "suggested mapping does not cover required local classes: "
            + ", ".join(missing_required)
        )

    return {
        "non_approving": True,
        "local_class_to_source_labels": {
            key: sorted(values)
            for key, values in suggested.items()
            if values and (key in required_classes or key in {"person", "safety_lanyard"})
        },
        "hard_negative_source_labels": sorted(set(hard_negatives)),
        "ignored_or_unmapped_source_labels": sorted(set(ignore_labels)),
        "warnings": warnings,
        "review_note": (
            "Reviewer must confirm or replace this suggestion in class_mapping before "
            "include_in_training=true. These hints are never treated as approval."
        ),
    }


def _source_review_queue_item(candidate: dict[str, Any]) -> dict[str, Any]:
    source_facts = _source_facts(candidate.get("source_facts"))
    source_research = _source_research_status(candidate)
    return {
        "review_priority": candidate.get("review_priority"),
        "source_ref": candidate.get("source_ref"),
        "capability": candidate.get("capability"),
        "url": candidate.get("url"),
        "checked": candidate.get("checked"),
        "license_note": candidate.get("license_note"),
        "source_counts": _source_count_summary(source_facts),
        "classes": _source_class_summary(source_facts),
        "suggested_mapping": candidate.get("suggested_mapping") or {},
        "legal_references": candidate.get("legal_references") or {},
        "source_research_status": source_research,
        "review_focus": candidate.get("review_focus"),
        "blocker": candidate.get("blocker"),
        "approval_status": candidate.get("approval_status"),
        "training_usable": candidate.get("training_usable") is True,
    }


def _source_research_status(candidate: dict[str, Any]) -> dict[str, Any]:
    disposition = str(candidate.get("source_research_disposition") or "open").strip()
    if disposition not in ALLOWED_SOURCE_RESEARCH_DISPOSITIONS:
        disposition = "open"
    evidence = (
        candidate.get("source_research_evidence")
        if isinstance(candidate.get("source_research_evidence"), list)
        else []
    )
    evidence_count = len(evidence)
    source_facts = _source_facts(candidate.get("source_facts"))
    has_source_facts = bool(source_facts)
    legal_references = candidate.get("legal_references") if isinstance(candidate.get("legal_references"), dict) else {}
    has_legal_references = bool(legal_references.get("license_reference_url") or legal_references.get("platform_references"))
    missing: list[str] = []
    if disposition == "unavailable_after_agent_search":
        if evidence_count < SOURCE_RESEARCH_EVIDENCE_READY_MIN_ITEMS:
            missing.append("source_research_evidence")
        research_closed = not missing
        return {
            "status": (
                "source_unavailable_after_agent_research"
                if research_closed
                else "source_unavailable_needs_agent_research"
            ),
            "evidence_ready_for_human_review": False,
            "agent_research_closed": research_closed,
            "source_unavailable": True,
            "source_research_disposition": disposition,
            "evidence_count": evidence_count,
            "minimum_evidence_items": SOURCE_RESEARCH_EVIDENCE_READY_MIN_ITEMS,
            "has_source_facts": has_source_facts,
            "has_legal_references": has_legal_references,
            "missing": missing,
            "next_action": (
                "keep_source_blocked_or_replace_with_verifiable_source"
                if research_closed
                else "record_enough_search_and_page_unavailability_evidence"
            ),
            "approval_note": (
                "This status closes agent research for an unavailable source. It does not approve "
                "the source for training, manifest import, or human/legal review."
            ),
        }
    if not has_source_facts:
        missing.append("source_facts")
    if evidence_count < SOURCE_RESEARCH_EVIDENCE_READY_MIN_ITEMS:
        missing.append("source_research_evidence")
    if not has_legal_references:
        missing.append("legal_references")
    evidence_ready = not missing
    if evidence_ready:
        status = "evidence_ready_for_human_review"
        next_action = "human_legal_review_and_seed_import_manifest"
    elif evidence_count:
        status = "partial_evidence_needs_agent_research"
        next_action = "complete_source_page_export_terms_and_license_evidence"
    else:
        status = "needs_agent_research"
        next_action = "collect_source_page_export_terms_and_license_evidence"
    return {
        "status": status,
        "evidence_ready_for_human_review": evidence_ready,
        "agent_research_closed": False,
        "source_unavailable": False,
        "source_research_disposition": disposition,
        "evidence_count": evidence_count,
        "minimum_evidence_items": SOURCE_RESEARCH_EVIDENCE_READY_MIN_ITEMS,
        "has_source_facts": has_source_facts,
        "has_legal_references": has_legal_references,
        "missing": missing,
        "next_action": next_action,
        "approval_note": (
            "This status only describes agent-collected research completeness. It does not approve "
            "the source for training or manifest import."
        ),
    }


def _is_reviewable_source_candidate(candidate: dict[str, Any]) -> bool:
    status = _source_research_status(candidate)
    return (
        status.get("evidence_ready_for_human_review") is True
        and status.get("source_unavailable") is not True
    )


def _source_research_readiness_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_candidates = sorted(candidates, key=_candidate_sort_key)
    evidence_ready = [
        candidate
        for candidate in sorted_candidates
        if _source_research_status(candidate)["evidence_ready_for_human_review"]
    ]
    research_closed = [
        candidate
        for candidate in sorted_candidates
        if (
            not _source_research_status(candidate)["evidence_ready_for_human_review"]
            and _source_research_status(candidate).get("agent_research_closed") is True
        )
    ]
    needs_research = [
        candidate
        for candidate in sorted_candidates
        if (
            not _source_research_status(candidate)["evidence_ready_for_human_review"]
            and _source_research_status(candidate).get("agent_research_closed") is not True
        )
    ]
    by_capability: dict[str, Any] = {}
    for capability in sorted(ALLOWED_CAPABILITIES):
        capability_candidates = [
            candidate for candidate in sorted_candidates if candidate.get("capability") == capability
        ]
        capability_ready = [
            candidate
            for candidate in capability_candidates
            if _source_research_status(candidate)["evidence_ready_for_human_review"]
        ]
        capability_closed = [
            candidate
            for candidate in capability_candidates
            if (
                not _source_research_status(candidate)["evidence_ready_for_human_review"]
                and _source_research_status(candidate).get("agent_research_closed") is True
            )
        ]
        capability_needs_research = [
            candidate
            for candidate in capability_candidates
            if (
                not _source_research_status(candidate)["evidence_ready_for_human_review"]
                and _source_research_status(candidate).get("agent_research_closed") is not True
            )
        ]
        by_capability[capability] = {
            "candidate_count": len(capability_candidates),
            "evidence_ready_count": len(capability_ready),
            "source_research_closed_count": len(capability_closed),
            "needs_agent_research_count": len(capability_needs_research),
            "evidence_ready_sources": [
                _source_review_queue_item(candidate) for candidate in capability_ready[:5]
            ],
            "source_research_closed_sources": [
                _source_review_queue_item(candidate) for candidate in capability_closed[:5]
            ],
            "next_agent_research_sources": [
                _source_review_queue_item(candidate) for candidate in capability_needs_research
            ][:5],
        }
    return {
        "candidate_count": len(sorted_candidates),
        "evidence_ready_count": len(evidence_ready),
        "needs_agent_research_count": len(needs_research),
        "source_research_closed_count": len(research_closed),
        "evidence_ready_unapproved_count": sum(
            1
            for candidate in evidence_ready
            if candidate.get("approval_status") != "approved_for_training"
        ),
        "training_usable_count": sum(
            1 for candidate in sorted_candidates if candidate.get("training_usable") is True
        ),
        "evidence_ready_sources": [
            _source_review_queue_item(candidate) for candidate in evidence_ready[:8]
        ],
        "source_research_closed_sources": [
            _source_review_queue_item(candidate) for candidate in research_closed[:8]
        ],
        "next_agent_research_sources": [
            _source_review_queue_item(candidate) for candidate in needs_research[:8]
        ],
        "capabilities": by_capability,
        "approval_guardrail": (
            "Evidence-ready means agent/browser research is sufficient to hand to a human/legal "
            "reviewer. It does not change approval_status, approved_for_training, training_usable, "
            "or include_in_training."
        ),
    }


def _source_review_queue_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_candidates = sorted(candidates, key=_candidate_sort_key)
    reviewable_candidates = [
        candidate for candidate in sorted_candidates if _is_reviewable_source_candidate(candidate)
    ]
    capability_summary: dict[str, Any] = {}
    for capability in sorted(ALLOWED_CAPABILITIES):
        capability_candidates = [
            candidate for candidate in sorted_candidates if candidate.get("capability") == capability
        ]
        capability_reviewable = [
            candidate
            for candidate in reviewable_candidates
            if candidate.get("capability") == capability
        ]
        capability_summary[capability] = {
            "candidate_count": len(capability_candidates),
            "reviewable_count": len(capability_reviewable),
            "training_usable_count": sum(
                1 for candidate in capability_candidates if candidate.get("training_usable") is True
            ),
            "unreviewed_count": sum(
                1
                for candidate in capability_candidates
                if candidate.get("approval_status") == "unreviewed"
            ),
            "top_review_candidates": [
                _source_review_queue_item(candidate)
                for candidate in capability_reviewable[:3]
            ],
        }
    return {
        "candidate_count": len(sorted_candidates),
        "reviewable_count": len(reviewable_candidates),
        "training_usable_count": sum(
            1 for candidate in sorted_candidates if candidate.get("training_usable") is True
        ),
        "unreviewed_count": sum(
            1 for candidate in sorted_candidates if candidate.get("approval_status") == "unreviewed"
        ),
        "next_review_sources": [
            _source_review_queue_item(candidate) for candidate in reviewable_candidates[:6]
        ],
        "capabilities": capability_summary,
        "review_packet_note": (
            "Review packets and evidence templates are generated separately; fill those files and "
            "the checklist before setting any source include_in_training=true."
        ),
    }


def build_review_queue(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a source-review queue joined with generated reviewer artifacts."""
    evidence_templates = _review_file_lookup(report, "review_evidence_templates")
    review_packets = _review_file_lookup(report, "review_packets")
    import_template = report.get("seed_import_manifest_template")
    if not isinstance(import_template, dict):
        import_template = {}
    checklist = report.get("review_checklist_csv")
    if not isinstance(checklist, dict):
        checklist = {}

    queue: list[dict[str, Any]] = []
    for candidate in sorted(report.get("candidates") or [], key=_candidate_sort_key):
        if not isinstance(candidate, dict):
            continue
        source_ref = str(candidate.get("source_ref") or "")
        capability = str(candidate.get("capability") or "")
        evidence = evidence_templates.get((source_ref, capability), {})
        packet = review_packets.get((source_ref, capability), {})
        prefill = _review_prefill_artifact(candidate)
        source_research_status = _source_research_status(candidate)
        queue.append(
            {
                **_source_review_queue_item(candidate),
                "required_review": list(candidate.get("required_review") or []),
                "completed_review": candidate.get("completed_review") or {},
                "blockers": list(candidate.get("blockers") or []),
                "errors": list(candidate.get("errors") or []),
                "warnings": list(candidate.get("warnings") or []),
                "seed_import_fill_plan": _seed_import_fill_plan(candidate),
                "review_artifacts": {
                    "review_packet_path": packet.get("path") or "",
                    "review_packet_sha256": packet.get("sha256") or "",
                    "review_evidence_template_path": evidence.get("path") or "",
                    "review_evidence_template_sha256": evidence.get("sha256") or "",
                    "review_prefill_path": prefill.get("path") or "",
                    "review_prefill_sha256": prefill.get("sha256") or "",
                    "review_checklist_csv_path": checklist.get("path") or "",
                    "review_checklist_csv_sha256": checklist.get("sha256") or "",
                    "seed_import_manifest_template_path": import_template.get("path") or "",
                    "seed_import_manifest_template_sha256": import_template.get("sha256") or "",
                },
                "next_action": (
                    "keep_source_blocked_or_replace_with_verifiable_source"
                    if source_research_status.get("source_unavailable") is True
                    else (
                        "fill_review_evidence_and_checklist_then_validate_seed_import_manifest"
                        if candidate.get("training_usable") is not True
                        else "fill_and_validate_seed_import_manifest"
                    )
                ),
            }
        )
    return queue


def _candidate_mapped_local_classes(candidate: dict[str, Any], required_classes: set[str]) -> dict[str, list[str]]:
    mapping = candidate.get("suggested_mapping") if isinstance(candidate.get("suggested_mapping"), dict) else {}
    local_mapping = (
        mapping.get("local_class_to_source_labels")
        if isinstance(mapping.get("local_class_to_source_labels"), dict)
        else {}
    )
    mapped: dict[str, list[str]] = {}
    for class_name in sorted(required_classes):
        labels = local_mapping.get(class_name)
        if isinstance(labels, list):
            normalized = [str(label).strip() for label in labels if str(label).strip()]
        elif labels:
            normalized = [str(labels).strip()]
        else:
            normalized = []
        if normalized:
            mapped[class_name] = normalized
    return mapped


def _coverage_source_counts(candidate: dict[str, Any]) -> dict[str, Any]:
    explicit = candidate.get("source_counts")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)
    return _source_count_summary(_source_facts(candidate.get("source_facts")))


def _coverage_source_classes(candidate: dict[str, Any]) -> list[str]:
    explicit = candidate.get("classes")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit]
    return _source_class_summary(_source_facts(candidate.get("source_facts")))


def _coverage_candidate_item(candidate: dict[str, Any], required_classes: set[str]) -> dict[str, Any]:
    mapped = _candidate_mapped_local_classes(candidate, required_classes)
    mapped_classes = sorted(mapped)
    person_required = "person" in required_classes
    return {
        "review_priority": candidate.get("review_priority"),
        "source_ref": candidate.get("source_ref"),
        "capability": candidate.get("capability"),
        "url": candidate.get("url"),
        "approval_status": candidate.get("approval_status"),
        "training_usable": candidate.get("training_usable") is True,
        "source_research_status": (
            candidate.get("source_research_status", {}).get("status")
            if isinstance(candidate.get("source_research_status"), dict)
            else ""
        ),
        "source_counts": _coverage_source_counts(candidate),
        "source_classes": _coverage_source_classes(candidate),
        "mapped_local_classes": mapped_classes,
        "local_class_to_source_labels": mapped,
        "missing_local_classes": sorted(required_classes - set(mapped_classes)),
        "person_box_status": (
            "not_required"
            if not person_required
            else "candidate_person_mapping_present"
            if "person" in mapped
            else "missing_person_mapping_requires_reviewed_person_box_reconciliation"
        ),
        "hard_negative_source_labels": (
            candidate.get("suggested_mapping", {}).get("hard_negative_source_labels", [])
            if isinstance(candidate.get("suggested_mapping"), dict)
            else []
        ),
        "mapping_warnings": (
            candidate.get("suggested_mapping", {}).get("warnings", [])
            if isinstance(candidate.get("suggested_mapping"), dict)
            else []
        ),
        "review_focus": candidate.get("review_focus") or "",
        "blocker": candidate.get("blocker") or "",
    }


def _person_box_reconciliation_plan(
    required_classes: set[str],
    coverage_candidates: list[dict[str, Any]],
    mapped_sources: dict[str, list[str]],
) -> dict[str, Any]:
    if "person" not in required_classes:
        return {
            "required": False,
            "status": "person_not_required_for_capability",
            "candidate_person_source_refs": [],
            "candidate_without_person_source_refs": [],
            "allowed_options": [],
            "not_allowed": [],
        }

    person_sources = sorted(source for source in mapped_sources.get("person", []) if source)
    missing_person_sources = [
        str(item.get("source_ref") or "")
        for item in coverage_candidates
        if "person" not in (item.get("mapped_local_classes") or [])
        and item.get("mapped_local_classes")
    ]
    missing_person_sources = sorted(source for source in missing_person_sources if source)
    return {
        "required": True,
        "status": (
            "candidate_person_mapping_present_pending_review"
            if person_sources
            else "missing_person_boxes_in_reviewable_suggested_mappings"
        ),
        "candidate_person_source_refs": person_sources,
        "candidate_without_person_source_refs": missing_person_sources,
        "allowed_options": [
            {
                "option": "controlled_capture",
                "requirement": (
                    "Capture commercially cleared apron/harness footage and manually approve "
                    "person plus target-PPE boxes in the capture manifest."
                ),
            },
            {
                "option": "manual_person_annotation_on_approved_seed_exports",
                "requirement": (
                    "Only after source approval and immutable export review, add person boxes "
                    "manually and record approved label-review rows before training."
                ),
            },
            {
                "option": "reviewed_auto_label_person_boxes",
                "requirement": (
                    "Auto-labeling can only prefill person boxes; every generated label must be "
                    "manually reviewed, tied to source_clip_id, and approved before import."
                ),
            },
        ],
        "not_allowed": [
            "treating apron-only or harness-only boxes as person coverage",
            "training from auto-labeled person boxes without manual review",
            "setting include_in_training=true before source review and import manifest validation",
        ],
    }


def _coverage_plan_for_capability(candidates: list[dict[str, Any]], capability: str) -> dict[str, Any]:
    required_classes = set(REQUIRED_IMPORT_COUNT_CLASSES.get(capability, set()))
    capability_candidates = [
        candidate
        for candidate in sorted(candidates, key=_candidate_sort_key)
        if candidate.get("capability") == capability and _is_reviewable_source_candidate(candidate)
    ]
    coverage_candidates = [
        _coverage_candidate_item(candidate, required_classes)
        for candidate in capability_candidates
    ]
    mapped_sources: dict[str, list[str]] = {class_name: [] for class_name in sorted(required_classes)}
    for item in coverage_candidates:
        for class_name in item["mapped_local_classes"]:
            mapped_sources.setdefault(class_name, []).append(str(item.get("source_ref") or ""))

    selected_sources: list[dict[str, Any]] = []
    covered_classes: set[str] = set()
    for item in coverage_candidates:
        newly_covered = sorted(set(item["mapped_local_classes"]) - covered_classes)
        if not newly_covered:
            continue
        selected_sources.append(
            {
                "source_ref": item.get("source_ref"),
                "review_priority": item.get("review_priority"),
                "newly_covered_classes": newly_covered,
                "mapped_local_classes": item["mapped_local_classes"],
                "missing_local_classes": item["missing_local_classes"],
                "training_usable": item["training_usable"],
                "approval_status": item["approval_status"],
            }
        )
        covered_classes.update(newly_covered)
        if covered_classes >= required_classes:
            break

    missing_across_candidates = sorted(
        class_name for class_name, sources in mapped_sources.items() if not sources
    )
    complete_single_sources = [
        str(item.get("source_ref") or "")
        for item in coverage_candidates
        if not item["missing_local_classes"]
    ]
    return {
        "required_local_classes": sorted(required_classes),
        "reviewable_source_count": len(coverage_candidates),
        "mapped_class_sources": mapped_sources,
        "missing_local_classes_across_reviewable_sources": missing_across_candidates,
        "complete_single_source_refs": complete_single_sources,
        "person_box_reconciliation": _person_box_reconciliation_plan(
            required_classes,
            coverage_candidates,
            mapped_sources,
        ),
        "priority_coverage_plan": {
            "selected_sources": selected_sources,
            "covered_classes": sorted(covered_classes),
            "missing_classes_after_priority_plan": sorted(required_classes - covered_classes),
            "status": (
                "candidate_coverage_complete_pending_review"
                if covered_classes >= required_classes
                else "candidate_coverage_gap_pending_capture_or_source_review"
            ),
        },
        "candidate_coverage": coverage_candidates,
        "approval_note": (
            "Coverage is derived from non-approving suggested mappings. It helps prioritize "
            "review, but it does not approve a source, verify labels, or permit training import."
        ),
    }


def build_source_coverage_plan(
    report: dict[str, Any],
    *,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    """Return a non-approving source coverage plan for apron/harness review."""
    candidates = [candidate for candidate in report.get("candidates") or [] if isinstance(candidate, dict)]
    by_capability = {
        capability: _coverage_plan_for_capability(candidates, capability)
        for capability in sorted(ALLOWED_CAPABILITIES)
    }
    required_target_classes = sorted(
        {class_name for classes in REQUIRED_IMPORT_COUNT_CLASSES.values() for class_name in classes}
    )
    combined_class_sources: dict[str, list[str]] = {class_name: [] for class_name in required_target_classes}
    for capability_plan in by_capability.values():
        mapped_sources = capability_plan.get("mapped_class_sources") or {}
        for class_name, sources in mapped_sources.items():
            if class_name not in combined_class_sources:
                combined_class_sources[class_name] = []
            for source_ref in sources or []:
                if source_ref not in combined_class_sources[class_name]:
                    combined_class_sources[class_name].append(source_ref)
    missing_target_classes = sorted(
        class_name for class_name, sources in combined_class_sources.items() if not sources
    )
    coverage_gap_count = sum(
        len(plan.get("missing_local_classes_across_reviewable_sources") or [])
        for plan in by_capability.values()
    )
    return {
        "kind": SOURCE_COVERAGE_PLAN_KIND,
        "version": 1,
        "generated_at": report.get("generated_at"),
        "source_review_report": source_review_report or "",
        "source_review_sha256": source_review_fingerprint(report),
        "source_review_status": report.get("status"),
        "source_review_gate_passed": report.get("gate_passed") is True,
        "candidate_count": report.get("candidate_count", 0),
        "training_usable_count": report.get("training_usable_count", 0),
        "target_local_classes": required_target_classes,
        "combined_mapped_class_sources": combined_class_sources,
        "missing_target_classes_across_reviewable_sources": missing_target_classes,
        "coverage_gap_count": coverage_gap_count,
        "capabilities": by_capability,
        "approval_guardrail": {
            "agent_can_collect_evidence": True,
            "agent_can_approve_training": False,
            "human_approval_required": True,
            "forbidden_fields": [
                "approved_for_training=true",
                "approval_status=approved_for_training",
                "include_in_training=true",
                "review_items.*.approved=true",
                "training_usable=true",
            ],
            "notes": (
                "This source coverage plan is a prioritization aid only. It does not verify "
                "dataset labels, approve source licenses, or authorize seed import/training."
            ),
        },
        "next_action": (
            "review_complete_coverage_sources_then_fill_checklist_and_seed_import_manifest"
            if not missing_target_classes and coverage_gap_count == 0
            else "review_best_sources_and_capture_or_source_missing_classes_before_training"
        ),
    }


def write_source_coverage_plan(
    report: dict[str, Any],
    out_path: Path,
    *,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan = build_source_coverage_plan(
        report,
        source_review_report=source_review_report,
    )
    out_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reconciliation_summary: dict[str, dict[str, Any]] = {}
    for capability, capability_plan in (plan.get("capabilities") or {}).items():
        if not isinstance(capability_plan, dict):
            continue
        reconciliation = capability_plan.get("person_box_reconciliation")
        if not isinstance(reconciliation, dict):
            continue
        reconciliation_summary[str(capability)] = {
            "status": reconciliation.get("status"),
            "candidate_person_source_count": len(reconciliation.get("candidate_person_source_refs") or []),
            "candidate_without_person_source_count": len(
                reconciliation.get("candidate_without_person_source_refs") or []
            ),
        }
    return {
        "path": _rel(out_path),
        "sha256": _sha256_file(out_path),
        "kind": SOURCE_COVERAGE_PLAN_KIND,
        "coverage_gap_count": plan["coverage_gap_count"],
        "candidate_count": plan["candidate_count"],
        "source_review_sha256": plan["source_review_sha256"],
        "person_box_reconciliation": reconciliation_summary,
    }


def _seed_import_fill_plan(candidate: dict[str, Any]) -> dict[str, Any]:
    capability = str(candidate.get("capability") or "")
    required_classes = sorted(REQUIRED_IMPORT_COUNT_CLASSES.get(capability, set()))
    suggested_mapping = (
        candidate.get("suggested_mapping")
        if isinstance(candidate.get("suggested_mapping"), dict)
        else {}
    )
    suggested_local_mapping = (
        suggested_mapping.get("local_class_to_source_labels")
        if isinstance(suggested_mapping.get("local_class_to_source_labels"), dict)
        else {}
    )
    reviewed_mapping_starter: dict[str, list[str]] = {}
    for class_name in required_classes:
        labels = suggested_local_mapping.get(class_name)
        if isinstance(labels, list):
            normalized = [str(label).strip() for label in labels if str(label).strip()]
        elif labels:
            normalized = [str(labels).strip()]
        else:
            normalized = []
        if normalized:
            reviewed_mapping_starter[class_name] = normalized
    missing_from_suggestion = sorted(set(required_classes) - set(reviewed_mapping_starter))
    return {
        "non_approving": True,
        "required_local_classes": list(required_classes),
        "reviewed_class_mapping_starter": reviewed_mapping_starter,
        "missing_required_classes_from_suggestion": missing_from_suggestion,
        "expected_count_classes_that_must_be_nonzero": list(required_classes),
        "required_fields_before_include_in_training": [
            "review_status=approved_for_training",
            "reviewed_by",
            "reviewed_at",
            "manifest_import_path",
            "raw_export_ref",
            "raw_export_sha256",
            "raw_export_local_path",
            "class_mapping",
            "person_box_policy",
            "hard_negative_policy",
            "split_plan.train",
            "split_plan.val",
            "split_plan.test",
            "expected_labeled_images_per_class",
        ],
        "reviewer_notes": [
            "This fill plan is derived from non-approving suggested mappings.",
            "Copy labels into class_mapping only after legal/export/provenance review and label inspection.",
            "Every class in expected_count_classes_that_must_be_nonzero must have a nonzero reviewed count.",
        ],
    }


def build_next_review_batch(
    report: dict[str, Any],
    *,
    limit: int = DEFAULT_NEXT_REVIEW_BATCH_LIMIT,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    """Return the next non-approving review batch for agent/browser source research."""
    queue = report.get("review_queue")
    if not isinstance(queue, list):
        queue = build_review_queue(report)
    effective_limit = max(int(limit or 0), 0)
    reviewable_queue = [
        item for item in queue
        if isinstance(item, dict)
        and isinstance(item.get("source_research_status"), dict)
        and item["source_research_status"].get("evidence_ready_for_human_review") is True
        and item["source_research_status"].get("source_unavailable") is not True
    ]
    selected_queue = reviewable_queue if effective_limit == 0 else reviewable_queue[:effective_limit]
    items: list[dict[str, Any]] = []
    for rank, item in enumerate(selected_queue, start=1):
        if not isinstance(item, dict):
            continue
        review_artifacts = (
            item.get("review_artifacts")
            if isinstance(item.get("review_artifacts"), dict)
            else {}
        )
        review_packet_path = review_artifacts.get("review_packet_path") or ""
        review_packet_sha256 = review_artifacts.get("review_packet_sha256") or ""
        review_evidence_template_path = review_artifacts.get("review_evidence_template_path") or ""
        review_evidence_template_sha256 = (
            review_artifacts.get("review_evidence_template_sha256") or ""
        )
        review_prefill_path = review_artifacts.get("review_prefill_path") or ""
        review_prefill_sha256 = review_artifacts.get("review_prefill_sha256") or ""
        review_checklist_csv_path = review_artifacts.get("review_checklist_csv_path") or ""
        review_checklist_csv_sha256 = review_artifacts.get("review_checklist_csv_sha256") or ""
        seed_import_manifest_template_path = (
            review_artifacts.get("seed_import_manifest_template_path") or ""
        )
        seed_import_manifest_template_sha256 = (
            review_artifacts.get("seed_import_manifest_template_sha256") or ""
        )
        items.append(
            {
                "rank": rank,
                "review_priority": item.get("review_priority"),
                "source_ref": item.get("source_ref"),
                "capability": item.get("capability"),
                "source_url": item.get("url"),
                "license_note": item.get("license_note"),
                "approval_status": item.get("approval_status"),
                "training_usable": item.get("training_usable") is True,
                "review_focus": item.get("review_focus") or "",
                "current_blocker": item.get("blocker") or "",
                "blockers": list(item.get("blockers") or []),
                "warnings": list(item.get("warnings") or []),
                "required_review": list(item.get("required_review") or []),
                "classes": list(item.get("classes") or []),
                "source_counts": item.get("source_counts") or {},
                "legal_references": item.get("legal_references") or {},
                "source_research_status": item.get("source_research_status") or {},
                "suggested_mapping": item.get("suggested_mapping") or {},
                "seed_import_fill_plan": _seed_import_fill_plan(item),
                "review_packet_path": review_packet_path,
                "review_packet_sha256": review_packet_sha256,
                "review_evidence_template_path": review_evidence_template_path,
                "review_evidence_template_sha256": review_evidence_template_sha256,
                "review_prefill_path": review_prefill_path,
                "review_prefill_sha256": review_prefill_sha256,
                "review_checklist_csv_path": review_checklist_csv_path,
                "review_checklist_csv_sha256": review_checklist_csv_sha256,
                "seed_import_manifest_template_path": seed_import_manifest_template_path,
                "seed_import_manifest_template_sha256": seed_import_manifest_template_sha256,
                "review_artifacts": {
                    "review_packet_path": review_packet_path,
                    "review_packet_sha256": review_packet_sha256,
                    "review_evidence_template_path": review_evidence_template_path,
                    "review_evidence_template_sha256": review_evidence_template_sha256,
                    "review_prefill_path": review_prefill_path,
                    "review_prefill_sha256": review_prefill_sha256,
                    "review_checklist_csv_path": review_checklist_csv_path,
                    "review_checklist_csv_sha256": review_checklist_csv_sha256,
                    "seed_import_manifest_template_path": seed_import_manifest_template_path,
                    "seed_import_manifest_template_sha256": seed_import_manifest_template_sha256,
                },
                "agent_research_tasks": AGENT_RESEARCH_TASKS,
                "human_approval_tasks": HUMAN_APPROVAL_TASKS,
                "next_action": item.get("next_action") or "",
            }
        )
    coverage_snapshot = _source_coverage_snapshot(
        report,
        source_review_report=source_review_report,
    )
    return {
        "kind": NEXT_REVIEW_BATCH_KIND,
        "version": 1,
        "generated_at": report.get("generated_at"),
        "source_review_report": source_review_report or "",
        "source_review_sha256": source_review_fingerprint(report),
        "source_review_status": report.get("status"),
        "source_review_gate_passed": report.get("gate_passed") is True,
        "source_review_inputs": report.get("inputs", {}),
        "candidate_count": report.get("candidate_count", 0),
        "training_usable_count": report.get("training_usable_count", 0),
        "source_research_readiness": report.get("source_research_readiness") or {},
        "source_coverage_snapshot": coverage_snapshot,
        "minimum_review_path": _minimum_review_path(coverage_snapshot, items),
        "limit": effective_limit,
        "selected_count": len(items),
        "approval_guardrail": {
            "agent_can_collect_evidence": True,
            "agent_can_approve_training": False,
            "human_approval_required": True,
            "forbidden_fields": [
                "approved_for_training=true",
                "approval_status=approved_for_training",
                "include_in_training=true",
                "review_items.*.approved=true",
                "training_usable=true",
            ],
            "notes": (
                "This batch is for source research and evidence collection only. It must not be "
                "used to approve a source for training or include any public export in training."
            ),
        },
        "items": items,
    }


def _source_coverage_snapshot(
    report: dict[str, Any],
    *,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    try:
        plan = build_source_coverage_plan(
            report,
            source_review_report=source_review_report,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "coverage_gap_count": None,
            "training_usable_count": report.get("training_usable_count", 0),
            "capabilities": {},
        }
    capabilities: dict[str, Any] = {}
    raw_capabilities = plan.get("capabilities") if isinstance(plan.get("capabilities"), dict) else {}
    for capability in sorted(ALLOWED_CAPABILITIES):
        raw = raw_capabilities.get(capability)
        if not isinstance(raw, dict):
            continue
        priority_plan = (
            raw.get("priority_coverage_plan")
            if isinstance(raw.get("priority_coverage_plan"), dict)
            else {}
        )
        selected_sources = [
            str(item.get("source_ref"))
            for item in priority_plan.get("selected_sources") or []
            if isinstance(item, dict) and item.get("source_ref")
        ]
        person_box = (
            raw.get("person_box_reconciliation")
            if isinstance(raw.get("person_box_reconciliation"), dict)
            else {}
        )
        capabilities[capability] = {
            "missing_local_classes": raw.get("missing_local_classes_across_reviewable_sources") or [],
            "person_box_status": person_box.get("status"),
            "priority_coverage_status": priority_plan.get("status"),
            "selected_sources": selected_sources,
        }
    return {
        "available": True,
        "coverage_gap_count": plan.get("coverage_gap_count"),
        "training_usable_count": plan.get("training_usable_count", 0),
        "capabilities": capabilities,
    }


def _minimum_review_path(
    coverage_snapshot: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the selected minimum source-review path as machine-readable handoff data."""
    capabilities = (
        coverage_snapshot.get("capabilities")
        if isinstance(coverage_snapshot.get("capabilities"), dict)
        else {}
    )
    selected_refs: list[str] = []
    capability_by_ref: dict[str, str] = {}
    for capability in sorted(capabilities):
        raw = capabilities.get(capability)
        if not isinstance(raw, dict):
            continue
        for source_ref in raw.get("selected_sources") or []:
            normalized = str(source_ref).strip()
            if not normalized:
                continue
            if normalized not in selected_refs:
                selected_refs.append(normalized)
            capability_by_ref.setdefault(normalized, str(capability))
    item_by_ref = {
        str(item.get("source_ref") or ""): item
        for item in items
        if isinstance(item, dict)
    }
    path_items: list[dict[str, Any]] = []
    for source_ref in selected_refs:
        item = item_by_ref.get(source_ref, {})
        path_items.append(
            {
                "source_ref": source_ref,
                "capability": item.get("capability") or capability_by_ref.get(source_ref, ""),
                "review_packet_path": item.get("review_packet_path") or "",
                "review_packet_sha256": item.get("review_packet_sha256") or "",
                "review_evidence_template_path": item.get("review_evidence_template_path") or "",
                "review_evidence_template_sha256": item.get("review_evidence_template_sha256") or "",
                "seed_import_manifest_template_path": (
                    "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml"
                ),
                "non_approving": True,
            }
        )
    return {
        "available": bool(path_items),
        "source_refs": selected_refs,
        "source_count": len(path_items),
        "approval_boundary": (
            "This minimum review path is a source-review prioritization aid only; it does not "
            "approve training, import public exports, or authorize customer-facing claims."
        ),
        "items": path_items,
    }


def write_next_review_batch(
    report: dict[str, Any],
    out_path: Path,
    *,
    limit: int = DEFAULT_NEXT_REVIEW_BATCH_LIMIT,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    """Write the next non-approving review batch and return artifact metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch = build_next_review_batch(
        report,
        limit=limit,
        source_review_report=source_review_report,
    )
    out_path.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    return {
        "path": _rel(out_path),
        "sha256": _sha256_file(out_path),
        "kind": batch["kind"],
        "limit": batch["limit"],
        "selected_count": batch["selected_count"],
    }


def validate_next_review_batch(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Validate the non-approving next-review batch and recorded artifact hashes."""
    errors: list[str] = []
    try:
        batch = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "path": _rel(path),
            "errors": [f"next review batch unreadable: {exc}"],
            "checked_artifact_count": 0,
        }
    if not isinstance(batch, dict):
        return {
            "ok": False,
            "path": _rel(path),
            "errors": ["next review batch must be a JSON object"],
            "checked_artifact_count": 0,
        }
    if batch.get("kind") != NEXT_REVIEW_BATCH_KIND:
        errors.append(f"kind must be {NEXT_REVIEW_BATCH_KIND}")
    if batch.get("version") != 1:
        errors.append("version must be 1")
    expected_fingerprint = source_review_fingerprint(report)
    if str(batch.get("source_review_sha256") or "") != expected_fingerprint:
        errors.append("source_review_sha256 does not match seed source review")
    guardrail = batch.get("approval_guardrail") if isinstance(batch.get("approval_guardrail"), dict) else {}
    if guardrail.get("agent_can_approve_training") is not False:
        errors.append("approval_guardrail.agent_can_approve_training must be false")
    forbidden_fields = set(guardrail.get("forbidden_fields") or [])
    for field in {
        "approved_for_training=true",
        "approval_status=approved_for_training",
        "include_in_training=true",
        "review_items.*.approved=true",
        "training_usable=true",
    }:
        if field not in forbidden_fields:
            errors.append(f"approval_guardrail.forbidden_fields missing {field}")
    items = batch.get("items") if isinstance(batch.get("items"), list) else []
    items = [item for item in items if isinstance(item, dict)]
    if int(batch.get("selected_count") or 0) != len(items):
        errors.append("selected_count must match items length")
    checked_artifact_count = 0
    for index, item in enumerate(items):
        context = f"items[{index}]"
        if not str(item.get("source_ref") or "").strip():
            errors.append(f"{context}.source_ref is required")
        if str(item.get("capability") or "") not in ALLOWED_CAPABILITIES:
            errors.append(f"{context}.capability must be one of {sorted(ALLOWED_CAPABILITIES)}")
        if item.get("training_usable") is True:
            errors.append(f"{context}.training_usable must remain false in non-approving batch")
        for prefix, path_field, sha_field in [
            ("review_packet", "review_packet_path", "review_packet_sha256"),
            ("review_evidence_template", "review_evidence_template_path", "review_evidence_template_sha256"),
            ("review_prefill", "review_prefill_path", "review_prefill_sha256"),
            ("review_checklist_csv", "review_checklist_csv_path", "review_checklist_csv_sha256"),
            ("seed_import_manifest_template", "seed_import_manifest_template_path", "seed_import_manifest_template_sha256"),
        ]:
            path_value = str(item.get(path_field) or "")
            sha256 = str(item.get(sha_field) or "")
            if not path_value and not sha256:
                continue
            checked_artifact_count += 1
            errors.extend(
                _bundle_artifact_file_errors(
                    {"path": path_value, "sha256": sha256},
                    context=f"{context}.{prefix}",
                )
            )
    minimum_path_errors, minimum_path_count = _validate_minimum_review_path(batch, report)
    errors.extend(minimum_path_errors)
    return {
        "ok": not errors,
        "path": _rel(path),
        "kind": batch.get("kind"),
        "source_review_sha256": batch.get("source_review_sha256"),
        "source_review_sha256_matches": str(batch.get("source_review_sha256") or "") == expected_fingerprint,
        "selected_count": len(items),
        "minimum_review_path_count": minimum_path_count,
        "checked_artifact_count": checked_artifact_count,
        "errors": errors,
    }


def _compact_list(values: Any, *, limit: int = 8) -> str:
    if not isinstance(values, list):
        return ""
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if len(normalized) > limit:
        normalized = [*normalized[:limit], f"+{len(normalized) - limit} more"]
    return ", ".join(normalized)


def _compact_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _compact_mapping(mapping: Any) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return "none"
    parts: list[str] = []
    for class_name, labels in sorted(mapping.items()):
        label_text = _compact_list(labels) if isinstance(labels, list) else str(labels or "")
        parts.append(f"{class_name}={label_text}")
    return "; ".join(parts)


def render_review_kickoff(batch: dict[str, Any]) -> str:
    """Render a concise, non-approving source-review kickoff from a batch JSON."""
    items = [item for item in batch.get("items") or [] if isinstance(item, dict)]
    lines: list[str] = [
        "# Apron/Harness Source Review Kickoff",
        "",
        f"Generated: {batch.get('generated_at') or ''}",
        "",
        "This is a non-approving operator handoff. It does not approve any source for "
        "training, manifest import, or customer-facing claims.",
        "",
        "## Gate Status",
        "",
        f"- Source review report: `{batch.get('source_review_report') or ''}`",
        f"- Source review SHA-256: `{batch.get('source_review_sha256') or ''}`",
        f"- Source review bundle: `{_rel(DEFAULT_REVIEW_BUNDLE)}`",
        f"- Source review status: `{batch.get('source_review_status') or ''}`",
        f"- Source review gate passed: `{bool(batch.get('source_review_gate_passed'))}`",
        f"- Candidate sources: `{batch.get('candidate_count', 0)}`",
        f"- Training-usable sources: `{batch.get('training_usable_count', 0)}`",
        f"- Selected review items: `{batch.get('selected_count', len(items))}`",
        "",
        "## Approval Guardrail",
        "",
        "- Agents can collect source evidence and prepare files.",
        "- Agents must not set approval, training, or include-in-training fields.",
        "- Human/legal review must sign off the evidence YAML, checklist row, and seed-import manifest.",
        "- If export access is paid, gated, customer-private, or unclear, stop and ask before using it.",
        "",
        "Forbidden before human/legal approval:",
        "",
    ]
    guardrail = batch.get("approval_guardrail") if isinstance(batch.get("approval_guardrail"), dict) else {}
    for field in guardrail.get("forbidden_fields") or []:
        lines.append(f"- `{field}`")
    coverage = (
        batch.get("source_coverage_snapshot")
        if isinstance(batch.get("source_coverage_snapshot"), dict)
        else {}
    )
    if coverage:
        lines.extend(
            [
                "",
                "## Source Coverage Snapshot",
                "",
                "This snapshot is built from non-approving suggested mappings. It prioritizes review work; it does not approve training import.",
                "",
                f"- Available: `{bool(coverage.get('available'))}`",
                f"- Coverage gaps: `{coverage.get('coverage_gap_count')}`",
                f"- Training-usable sources: `{coverage.get('training_usable_count', 0)}`",
                "",
                "| Capability | Missing Classes | Person Boxes | Priority Coverage | Selected Sources |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        capabilities = coverage.get("capabilities") if isinstance(coverage.get("capabilities"), dict) else {}
        if capabilities:
            for capability in sorted(capabilities):
                item = capabilities.get(capability)
                if not isinstance(item, dict):
                    continue
                missing = ", ".join(str(value) for value in item.get("missing_local_classes") or []) or "none"
                selected = ", ".join(str(value) for value in item.get("selected_sources") or []) or "none"
                lines.append(
                    "| "
                    f"`{_md_cell(capability)}` | "
                    f"{_md_cell(missing)} | "
                    f"`{_md_cell(item.get('person_box_status'))}` | "
                    f"`{_md_cell(item.get('priority_coverage_status'))}` | "
                    f"{_md_cell(selected)} |"
                )
        else:
            lines.append("| none | unknown | unknown | unknown | none |")
        minimum_refs: list[str] = []
        if capabilities:
            for capability in sorted(capabilities):
                item = capabilities.get(capability)
                if not isinstance(item, dict):
                    continue
                for source_ref in item.get("selected_sources") or []:
                    normalized = str(source_ref).strip()
                    if normalized and normalized not in minimum_refs:
                        minimum_refs.append(normalized)
        if minimum_refs:
            item_by_ref = {
                str(item.get("source_ref") or ""): item
                for item in items
                if isinstance(item, dict)
            }
            lines.extend(
                [
                    "",
                    "## Minimum Review Path",
                    "",
                    "Review these selected sources first. They are the shortest non-approving path to apron/harness coverage, not approval to train.",
                    "",
                    "| Source | Capability | Packet | Evidence Template | Seed Import Template |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for source_ref in minimum_refs:
                item = item_by_ref.get(source_ref, {})
                lines.append(
                    "| "
                    f"`{_md_cell(source_ref)}` | "
                    f"`{_md_cell(item.get('capability'))}` | "
                    f"`{_md_cell(item.get('review_packet_path'))}` | "
                    f"`{_md_cell(item.get('review_evidence_template_path'))}` | "
                    "`qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml` |"
                )
    lines.extend(
        [
            "",
            "## Start With This Batch",
            "",
            "| Rank | Capability | Source | URL | License Note | Counts | Key Classes | Mapping Gaps | Packet | Evidence Template | Prefill Memo | Review Focus |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in items:
        mapping = item.get("suggested_mapping") if isinstance(item.get("suggested_mapping"), dict) else {}
        warnings = _compact_list(mapping.get("warnings") or [])
        if not warnings:
            warnings = "none from non-approving hint"
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item.get("rank")),
                    f"`{_md_cell(item.get('capability'))}`",
                    f"`{_md_cell(item.get('source_ref'))}`",
                    _md_cell(item.get("source_url")),
                    _md_cell(item.get("license_note")),
                    _md_cell(_compact_counts(item.get("source_counts"))),
                    _md_cell(_compact_list(item.get("classes") or [])),
                    _md_cell(warnings),
                    f"`{_md_cell(item.get('review_packet_path'))}`",
                    f"`{_md_cell(item.get('review_evidence_template_path'))}`",
                    f"`{_md_cell(item.get('review_prefill_path'))}`",
                    _md_cell(item.get("review_focus")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Seed Import Fill Plan",
            "",
            "This plan is still non-approving. It shows what a reviewed seed-import row must fill after legal/export/provenance review.",
            "",
            "| Rank | Source | Required Local Classes | Mapping Starter | Missing From Suggestion | Required Nonzero Counts | Required Fields |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in items:
        fill_plan = (
            item.get("seed_import_fill_plan")
            if isinstance(item.get("seed_import_fill_plan"), dict)
            else {}
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item.get("rank")),
                    f"`{_md_cell(item.get('source_ref'))}`",
                    _md_cell(_compact_list(fill_plan.get("required_local_classes") or [])),
                    _md_cell(_compact_mapping(fill_plan.get("reviewed_class_mapping_starter"))),
                    _md_cell(
                        _compact_list(fill_plan.get("missing_required_classes_from_suggestion") or [])
                        or "none"
                    ),
                    _md_cell(
                        _compact_list(
                            fill_plan.get("expected_count_classes_that_must_be_nonzero") or []
                        )
                    ),
                    _md_cell(
                        _compact_list(
                            fill_plan.get("required_fields_before_include_in_training") or [],
                            limit=6,
                        )
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## YOLO Export ZIP Proof",
            "",
            "A source approval is not enough to import seed data. Before any row can set `include_in_training=true`, the reviewed seed-import manifest must point to a local YOLO export ZIP and validation must prove:",
            "",
            "- `raw_export_ref` is a remote immutable export reference, not a workstation path.",
            "- `raw_export_sha256` matches the reviewed local ZIP at `raw_export_local_path`.",
            "- The ZIP is readable and contains `data.yaml` plus train/valid/test image and label folders.",
            "- Every label file has a matching image file; orphan labels block import.",
            "- `class_mapping` covers every required local class listed in the fill plan.",
            "- Every class in `expected_count_classes_that_must_be_nonzero` has nonzero mapped YOLO label-file counts.",
            "- Review artifact paths and SHA-256 values still match the generated source-review bundle.",
            "",
            "## Required Steps",
            "",
            "1. Validate the next-review batch and source-review bundle hashes before using generated packets or evidence templates.",
            "2. Open the packet and prefill memo for each source, then archive the visible source facts, terms, and export evidence.",
            "3. Fill a copy of the matching review evidence YAML. Keep `approved: false` until the human/legal reviewer signs off.",
            "4. Update the matching row in `qa/video_eval/results/apron_harness_seed_source_review_checklist.csv` with reviewer, timestamp, evidence path, and SHA-256.",
            "5. Start with `qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml` for the priority coverage path, or use the full `qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml` only when supplemental sources are approved for training.",
            "6. Validate the filled checklist/import manifest with the commands below before any public seed data enters training.",
            "",
            "```bash",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
            "  --validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json",
            "",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
            "  --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
            "",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
            "  --model-packs qa/video_eval/model_packs.yaml \\",
            "  --apply-review-checklist-csv /path/to/filled/apron_harness_seed_source_review_checklist.csv \\",
            "  --updated-model-packs-out /path/to/reviewed/model_packs.yaml",
            "",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
            "  --model-packs /path/to/reviewed/model_packs.yaml \\",
            "  --validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml",
            "```",
            "",
            "The filled import-manifest command must exit `0` and print `IMPORT_MANIFEST: gate=pass` before any public seed export is materialized. A template or partially filled manifest must remain blocked and exit nonzero with `IMPORT_MANIFEST: gate=blocked`.",
            "",
            "## Production Reminder",
            "",
            "Even after source approval, production PPE compliance remains blocked until reviewed labels, "
            "closed-set YOLO26n/s training, side-by-side YAML runtime tests, and the Jetson 3-camera "
            "gate all pass.",
            "",
        ]
    )
    return "\n".join(lines)


def write_review_kickoff(
    report: dict[str, Any],
    out_path: Path,
    *,
    limit: int = DEFAULT_NEXT_REVIEW_BATCH_LIMIT,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    """Write a concise Markdown kickoff for the next non-approving review batch."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch = build_next_review_batch(
        report,
        limit=limit,
        source_review_report=source_review_report,
    )
    out_path.write_text(render_review_kickoff(batch), encoding="utf-8")
    return {
        "path": _rel(out_path),
        "sha256": _sha256_file(out_path),
        "kind": REVIEW_KICKOFF_KIND,
        "limit": batch["limit"],
        "selected_count": batch["selected_count"],
        "source_review_sha256": batch["source_review_sha256"],
    }


def _artifact_entry(metadata: Any, *, kind: str) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    path = str(metadata.get("path") or "").strip()
    sha256 = str(metadata.get("sha256") or "").strip()
    if not path and not sha256:
        return None
    return {
        "kind": kind,
        "path": path,
        "sha256": sha256,
    }


def _artifact_collection(metadata: Any, *, kind: str) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {"kind": kind, "dir": "", "count": 0, "files": []}
    files = [
        {
            "source_ref": item.get("source_ref") or "",
            "capability": item.get("capability") or "",
            "path": item.get("path") or "",
            "sha256": item.get("sha256") or "",
        }
        for item in metadata.get("files") or []
        if isinstance(item, dict)
    ]
    return {
        "kind": kind,
        "dir": metadata.get("dir") or "",
        "count": len(files),
        "files": files,
    }


def _review_prefill_collection(report: dict[str, Any]) -> dict[str, Any]:
    queue = report.get("review_queue")
    if not isinstance(queue, list):
        queue = build_review_queue(report)
    files: list[dict[str, Any]] = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        review_artifacts = (
            item.get("review_artifacts")
            if isinstance(item.get("review_artifacts"), dict)
            else {}
        )
        path = str(
            review_artifacts.get("review_prefill_path")
            or item.get("review_prefill_path")
            or ""
        ).strip()
        sha256 = str(
            review_artifacts.get("review_prefill_sha256")
            or item.get("review_prefill_sha256")
            or ""
        ).strip()
        if not path and not sha256:
            continue
        files.append(
            {
                "source_ref": item.get("source_ref") or "",
                "capability": item.get("capability") or "",
                "path": path,
                "sha256": sha256,
            }
        )
    return {
        "kind": "review_prefills",
        "dir": _rel(DEFAULT_REVIEW_PREFILL_DIR),
        "count": len(files),
        "files": files,
    }


def build_review_bundle_manifest(
    report: dict[str, Any],
    *,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    """Return a self-contained, non-approving handoff manifest for source review."""
    minimum_batch = build_next_review_batch(
        report,
        limit=0,
        source_review_report=source_review_report,
    )
    single_artifacts = [
        entry
        for entry in [
            _artifact_entry(report.get("work_order"), kind="source_review_work_order"),
            _artifact_entry(report.get("review_checklist_csv"), kind="review_checklist_csv"),
            _artifact_entry(
                report.get("seed_import_manifest_template"),
                kind="seed_import_manifest_template",
            ),
            _artifact_entry(
                report.get("minimum_seed_import_manifest_template"),
                kind="minimum_seed_import_manifest_template",
            ),
            _artifact_entry(report.get("next_review_batch"), kind="next_review_batch"),
            _artifact_entry(report.get("review_kickoff"), kind="review_kickoff"),
            _artifact_entry(report.get("source_coverage_plan"), kind="source_coverage_plan"),
            _artifact_entry(report.get("source_recheck"), kind="source_recheck"),
        ]
        if entry is not None
    ]
    return {
        "kind": REVIEW_BUNDLE_KIND,
        "version": 1,
        "generated_at": report.get("generated_at"),
        "source_review_report": source_review_report or "",
        "source_review_sha256": source_review_fingerprint(report),
        "source_review_status": report.get("status"),
        "source_review_gate_passed": report.get("gate_passed") is True,
        "candidate_count": report.get("candidate_count", 0),
        "training_usable_count": report.get("training_usable_count", 0),
        "review_queue_summary": report.get("review_queue_summary") or {},
        "source_research_readiness": report.get("source_research_readiness") or {},
        "minimum_review_path": minimum_batch.get("minimum_review_path") or {},
        "approval_guardrail": {
            "agent_can_collect_evidence": True,
            "agent_can_approve_training": False,
            "human_approval_required": True,
            "forbidden_fields": [
                "approved_for_training=true",
                "approval_status=approved_for_training",
                "include_in_training=true",
                "review_items.*.approved=true",
                "training_usable=true",
            ],
            "notes": (
                "This bundle verifies source-review handoff artifacts only. It does not "
                "approve a source for training or authorize importing any public export."
            ),
        },
        "artifacts": {
            "single_files": single_artifacts,
            "review_evidence_templates": _artifact_collection(
                report.get("review_evidence_templates"),
                kind="review_evidence_templates",
            ),
            "review_packets": _artifact_collection(
                report.get("review_packets"),
                kind="review_packets",
            ),
            "review_prefills": _review_prefill_collection(report),
        },
        "verification_commands": [
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
            "--out qa/video_eval/results/apron_harness_seed_source_review.json "
            "--work-order-out qa/video_eval/results/apron_harness_seed_source_review.md "
            "--import-template-out qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml "
            "--minimum-import-template-out qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml "
            "--review-checklist-csv-out qa/video_eval/results/apron_harness_seed_source_review_checklist.csv "
            "--review-evidence-template-dir qa/video_eval/results/apron_harness_seed_source_review_evidence "
            "--review-packet-dir qa/video_eval/results/apron_harness_seed_source_review_packets "
            "--next-review-batch-out qa/video_eval/results/apron_harness_next_source_review_batch.json "
            "--review-kickoff-out qa/video_eval/results/apron_harness_source_review_kickoff.md "
            "--source-coverage-plan-out qa/video_eval/results/apron_harness_source_coverage_plan.json "
            "--review-bundle-out qa/video_eval/results/apron_harness_source_review_bundle.json",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
            "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
            "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
            "--model-packs qa/video_eval/model_packs.yaml "
            "--apply-review-checklist-csv /path/to/filled/apron_harness_seed_source_review_checklist.csv "
            "--updated-model-packs-out /path/to/reviewed/model_packs.yaml",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
            "--model-packs /path/to/reviewed/model_packs.yaml "
            "--validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml",
        ],
    }


def write_review_bundle_manifest(
    report: dict[str, Any],
    out_path: Path,
    *,
    source_review_report: str | None = None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_review_bundle_manifest(
        report,
        source_review_report=source_review_report,
    )
    out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return {
        "path": _rel(out_path),
        "sha256": _sha256_file(out_path),
        "kind": REVIEW_BUNDLE_KIND,
        "review_packet_count": bundle["artifacts"]["review_packets"]["count"],
        "review_evidence_template_count": bundle["artifacts"]["review_evidence_templates"]["count"],
        "review_prefill_count": bundle["artifacts"]["review_prefills"]["count"],
        "source_review_sha256": bundle["source_review_sha256"],
    }


def _review_bundle_expected_single_artifacts(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in [
            _artifact_entry(report.get("work_order"), kind="source_review_work_order"),
            _artifact_entry(report.get("review_checklist_csv"), kind="review_checklist_csv"),
            _artifact_entry(
                report.get("seed_import_manifest_template"),
                kind="seed_import_manifest_template",
            ),
            _artifact_entry(
                report.get("minimum_seed_import_manifest_template"),
                kind="minimum_seed_import_manifest_template",
            ),
            _artifact_entry(report.get("next_review_batch"), kind="next_review_batch"),
            _artifact_entry(report.get("review_kickoff"), kind="review_kickoff"),
            _artifact_entry(report.get("source_coverage_plan"), kind="source_coverage_plan"),
            _artifact_entry(report.get("source_recheck"), kind="source_recheck"),
        ]
        if entry is not None
    ]


def _bundle_artifact_file_errors(entry: dict[str, Any], *, context: str) -> list[str]:
    errors: list[str] = []
    path_value = str(entry.get("path") or "").strip()
    sha256 = str(entry.get("sha256") or "").strip()
    if not path_value:
        errors.append(f"{context}.path is required")
        return errors
    if "://" in path_value:
        errors.append(f"{context}.path must be a local generated artifact path, not a URL")
        return errors
    path = _resolve_local_path(path_value)
    if path is None or not path.is_file():
        errors.append(f"{context}.path must point to an existing generated artifact file")
        return errors
    sha_error = _sha256_error(sha256, f"{context}.sha256")
    if sha_error:
        errors.append(sha_error)
        return errors
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != sha256:
        errors.append(f"{context}.sha256 does not match generated artifact file")
    return errors


def _bundle_collection_entries(bundle: dict[str, Any], key: str) -> list[dict[str, Any]]:
    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), dict) else {}
    collection = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
    files = collection.get("files") if isinstance(collection.get("files"), list) else []
    return [item for item in files if isinstance(item, dict)]


def _report_collection_entries(report: dict[str, Any], key: str) -> list[dict[str, Any]]:
    collection = report.get(key) if isinstance(report.get(key), dict) else {}
    files = collection.get("files") if isinstance(collection.get("files"), list) else []
    return [item for item in files if isinstance(item, dict)]


def _artifact_identity(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("source_ref") or ""),
        str(item.get("capability") or ""),
        str(item.get("path") or ""),
    )


def _validate_bundle_collection(
    *,
    bundle: dict[str, Any],
    report: dict[str, Any],
    key: str,
    expected_report_key: str,
    expected_items: list[dict[str, Any]] | None = None,
) -> tuple[list[str], int]:
    errors: list[str] = []
    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), dict) else {}
    collection = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
    files = _bundle_collection_entries(bundle, key)
    if int(collection.get("count") or 0) != len(files):
        errors.append(f"artifacts.{key}.count must match files length")
    for index, item in enumerate(files):
        context = f"artifacts.{key}.files[{index}]"
        if not str(item.get("source_ref") or "").strip():
            errors.append(f"{context}.source_ref is required")
        if str(item.get("capability") or "") not in ALLOWED_CAPABILITIES:
            errors.append(f"{context}.capability must be one of {sorted(ALLOWED_CAPABILITIES)}")
        errors.extend(_bundle_artifact_file_errors(item, context=context))

    if expected_items is None:
        expected_items = _report_collection_entries(report, expected_report_key)
    if expected_items:
        actual_by_identity = {_artifact_identity(item): item for item in files}
        for expected in expected_items:
            identity = _artifact_identity(expected)
            actual = actual_by_identity.get(identity)
            if actual is None:
                errors.append(
                    f"artifacts.{key} missing current generated artifact "
                    f"{expected.get('source_ref')}/{expected.get('capability')}: {expected.get('path')}"
                )
                continue
            if str(actual.get("sha256") or "") != str(expected.get("sha256") or ""):
                errors.append(
                    f"artifacts.{key} SHA-256 mismatch for "
                    f"{expected.get('source_ref')}/{expected.get('capability')}"
                )
    return errors, len(files)


def _validate_minimum_review_path(
    bundle: dict[str, Any],
    report: dict[str, Any],
) -> tuple[list[str], int]:
    errors: list[str] = []
    actual = bundle.get("minimum_review_path")
    if not isinstance(actual, dict):
        return ["minimum_review_path must be present"], 0
    expected_batch = build_next_review_batch(report, limit=0)
    expected = expected_batch.get("minimum_review_path") or {}
    expected_refs = list(expected.get("source_refs") or [])
    actual_refs = list(actual.get("source_refs") or [])
    if actual_refs != expected_refs:
        errors.append("minimum_review_path.source_refs must match current source coverage plan")
    actual_items = actual.get("items") if isinstance(actual.get("items"), list) else []
    actual_items = [item for item in actual_items if isinstance(item, dict)]
    if int(actual.get("source_count") or 0) != len(actual_items):
        errors.append("minimum_review_path.source_count must match items length")
    if int(actual.get("source_count") or 0) != len(expected_refs):
        errors.append("minimum_review_path.source_count must match selected source refs")
    if not str(actual.get("approval_boundary") or "").strip():
        errors.append("minimum_review_path.approval_boundary is required")
    actual_by_ref = {str(item.get("source_ref") or ""): item for item in actual_items}
    expected_items = expected.get("items") if isinstance(expected.get("items"), list) else []
    for expected_item in expected_items:
        if not isinstance(expected_item, dict):
            continue
        source_ref = str(expected_item.get("source_ref") or "")
        item = actual_by_ref.get(source_ref)
        if item is None:
            errors.append(f"minimum_review_path.items missing {source_ref}")
            continue
        if item.get("non_approving") is not True:
            errors.append(f"minimum_review_path.items.{source_ref}.non_approving must be true")
        if str(item.get("capability") or "") != str(expected_item.get("capability") or ""):
            errors.append(f"minimum_review_path.items.{source_ref}.capability does not match current coverage plan")
        for field in [
            "review_packet_path",
            "review_packet_sha256",
            "review_evidence_template_path",
            "review_evidence_template_sha256",
            "seed_import_manifest_template_path",
        ]:
            expected_value = str(expected_item.get(field) or "")
            if expected_value and str(item.get(field) or "") != expected_value:
                errors.append(f"minimum_review_path.items.{source_ref}.{field} does not match current handoff")
        for prefix, path_field, sha_field in [
            ("review_packet", "review_packet_path", "review_packet_sha256"),
            ("review_evidence_template", "review_evidence_template_path", "review_evidence_template_sha256"),
        ]:
            path_value = str(item.get(path_field) or "")
            sha256 = str(item.get(sha_field) or "")
            artifact_errors = _bundle_artifact_file_errors(
                {"path": path_value, "sha256": sha256},
                context=f"minimum_review_path.items.{source_ref}.{prefix}",
            )
            errors.extend(artifact_errors)
    return errors, len(actual_items)


def validate_review_bundle_manifest(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Validate the non-approving source-review handoff bundle and artifact hashes."""
    errors: list[str] = []
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "path": _rel(path),
            "errors": [f"review bundle unreadable: {exc}"],
            "checked_artifact_count": 0,
        }
    if not isinstance(bundle, dict):
        return {
            "ok": False,
            "path": _rel(path),
            "errors": ["review bundle must be a JSON object"],
            "checked_artifact_count": 0,
        }

    if bundle.get("kind") != REVIEW_BUNDLE_KIND:
        errors.append(f"kind must be {REVIEW_BUNDLE_KIND}")
    if bundle.get("version") != 1:
        errors.append("version must be 1")
    expected_fingerprint = source_review_fingerprint(report)
    if str(bundle.get("source_review_sha256") or "") != expected_fingerprint:
        errors.append("source_review_sha256 does not match seed source review")
    if bool(bundle.get("source_review_gate_passed")) != (report.get("gate_passed") is True):
        errors.append("source_review_gate_passed does not match seed source review")
    if int(bundle.get("candidate_count") or 0) != int(report.get("candidate_count") or 0):
        errors.append("candidate_count does not match seed source review")
    if int(bundle.get("training_usable_count") or 0) != int(report.get("training_usable_count") or 0):
        errors.append("training_usable_count does not match seed source review")

    guardrail = bundle.get("approval_guardrail") if isinstance(bundle.get("approval_guardrail"), dict) else {}
    if guardrail.get("agent_can_approve_training") is not False:
        errors.append("approval_guardrail.agent_can_approve_training must be false")
    forbidden_fields = set(guardrail.get("forbidden_fields") or [])
    for field in {
        "approved_for_training=true",
        "approval_status=approved_for_training",
        "include_in_training=true",
        "review_items.*.approved=true",
        "training_usable=true",
    }:
        if field not in forbidden_fields:
            errors.append(f"approval_guardrail.forbidden_fields missing {field}")

    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), dict) else {}
    single_files = artifacts.get("single_files") if isinstance(artifacts.get("single_files"), list) else []
    single_files = [item for item in single_files if isinstance(item, dict)]
    for index, item in enumerate(single_files):
        if not str(item.get("kind") or "").strip():
            errors.append(f"artifacts.single_files[{index}].kind is required")
        errors.extend(_bundle_artifact_file_errors(item, context=f"artifacts.single_files[{index}]"))

    expected_singles = _review_bundle_expected_single_artifacts(report)
    actual_singles_by_kind = {str(item.get("kind") or ""): item for item in single_files}
    for expected in expected_singles:
        kind = str(expected.get("kind") or "")
        actual = actual_singles_by_kind.get(kind)
        if actual is None:
            errors.append(f"artifacts.single_files missing {kind}")
            continue
        if str(actual.get("path") or "") != str(expected.get("path") or ""):
            errors.append(f"artifacts.single_files.{kind}.path does not match current generated artifact")
        if str(actual.get("sha256") or "") != str(expected.get("sha256") or ""):
            errors.append(f"artifacts.single_files.{kind}.sha256 does not match current generated artifact")

    packet_errors, packet_count = _validate_bundle_collection(
        bundle=bundle,
        report=report,
        key="review_packets",
        expected_report_key="review_packets",
    )
    template_errors, template_count = _validate_bundle_collection(
        bundle=bundle,
        report=report,
        key="review_evidence_templates",
        expected_report_key="review_evidence_templates",
    )
    prefill_errors, prefill_count = _validate_bundle_collection(
        bundle=bundle,
        report=report,
        key="review_prefills",
        expected_report_key="",
        expected_items=_review_prefill_collection(report)["files"],
    )
    errors.extend(packet_errors)
    errors.extend(template_errors)
    errors.extend(prefill_errors)
    minimum_path_errors, minimum_path_count = _validate_minimum_review_path(bundle, report)
    errors.extend(minimum_path_errors)

    return {
        "ok": not errors,
        "path": _rel(path),
        "kind": bundle.get("kind"),
        "source_review_sha256": bundle.get("source_review_sha256"),
        "source_review_sha256_matches": str(bundle.get("source_review_sha256") or "") == expected_fingerprint,
        "source_review_gate_passed": bundle.get("source_review_gate_passed") is True,
        "single_file_count": len(single_files),
        "review_packet_count": packet_count,
        "review_evidence_template_count": template_count,
        "review_prefill_count": prefill_count,
        "minimum_review_path_count": minimum_path_count,
        "checked_artifact_count": len(single_files) + packet_count + template_count + prefill_count,
        "errors": errors,
    }


def _md_cell(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|").strip()


def _legal_references_for_source(source: dict[str, Any]) -> dict[str, Any]:
    url = str(source.get("url") or "")
    license_note = str(source.get("license_note") or "")
    license_url = None
    for license_name, reference_url in LICENSE_REFERENCE_URLS.items():
        if license_name in license_note:
            license_url = reference_url
            break
    return {
        "source_url": url,
        "license_note": license_note,
        "license_reference_url": license_url,
        "platform_references": dict(ROBOFLOW_PLATFORM_REFERENCES)
        if "roboflow.com" in url
        else {},
        "review_note": (
            "Confirm the source page license, Roboflow Terms, export permissions, attribution "
            "requirements, dataset provenance, and commercial training fit before approval."
        ),
    }


def _source_facts(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    slug = "".join(char if char.isalnum() else "_" for char in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _prune_stale_generated_files(directory: Path, pattern: str, keep_filenames: set[str]) -> None:
    for path in directory.glob(pattern):
        if path.name not in keep_filenames and path.is_file():
            path.unlink()


def _raw_export_ref_error(raw_export_ref: Any) -> str | None:
    ref = str(raw_export_ref or "").strip()
    if not ref:
        return "raw_export_ref is required when include_in_training=true"
    if "://" not in ref:
        return "raw_export_ref must be a remote immutable export reference, not a local path"
    scheme = ref.split("://", 1)[0].lower()
    if scheme not in ALLOWED_RAW_EXPORT_REF_SCHEMES:
        return (
            "raw_export_ref scheme must be one of "
            + ", ".join(sorted(ALLOWED_RAW_EXPORT_REF_SCHEMES))
        )
    if scheme == "http":
        return "raw_export_ref must use https or object storage, not http"
    return None


def _raw_export_sha256_error(raw_export_sha256: Any) -> str | None:
    digest = str(raw_export_sha256 or "").strip()
    if not digest:
        return "raw_export_sha256 is required when include_in_training=true"
    if len(digest) != 64 or any(char not in HEX_DIGITS for char in digest):
        return "raw_export_sha256 must be a 64-character SHA-256 hex digest"
    return None


def _sha256_error(value: Any, field: str) -> str | None:
    digest = str(value or "").strip()
    if not digest:
        return f"{field} is required"
    if len(digest) != 64 or any(char not in HEX_DIGITS for char in digest):
        return f"{field} must be a 64-character SHA-256 hex digest"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_artifact_path(path_value: Any) -> tuple[Path | None, list[str]]:
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return None, ["raw_export_local_path is required when include_in_training=true"]
    if "://" in raw_path:
        return None, ["raw_export_local_path must be a local file path, not a URL"]
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return path, ["raw_export_local_path must point to an existing local export archive"]
    return path, []


def _resolve_local_path(path_value: Any) -> Path | None:
    raw_path = str(path_value or "").strip()
    if not raw_path or "://" in raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _review_artifact_field_errors(
    review_artifacts: dict[str, Any],
    expected: dict[str, Any],
    *,
    path_field: str,
    sha256_field: str,
) -> list[str]:
    errors: list[str] = []
    expected_path = str(expected.get("path") or "").strip()
    expected_sha256 = str(expected.get("sha256") or "").strip()
    if not expected_path and not expected_sha256:
        return errors

    actual_path = str(review_artifacts.get(path_field) or "").strip()
    actual_sha256 = str(review_artifacts.get(sha256_field) or "").strip()
    dotted_path = f"review_artifacts.{path_field}"
    dotted_sha256 = f"review_artifacts.{sha256_field}"

    if expected_path:
        if not actual_path:
            errors.append(f"{dotted_path} is required when seed source review has generated artifacts")
        elif "://" in actual_path:
            errors.append(f"{dotted_path} must be a local generated artifact path, not a URL")
        else:
            actual_resolved = _resolve_local_path(actual_path)
            expected_resolved = _resolve_local_path(expected_path)
            if actual_path != expected_path and actual_resolved != expected_resolved:
                errors.append(f"{dotted_path} must match the seed source review artifact path")
            elif actual_resolved is None or not actual_resolved.is_file():
                errors.append(f"{dotted_path} must point to an existing generated artifact file")
            elif expected_sha256 and _sha256_file(actual_resolved) != expected_sha256:
                errors.append(f"{dotted_sha256} does not match the generated artifact file")

    if expected_sha256:
        sha_error = _sha256_error(actual_sha256, dotted_sha256)
        if sha_error:
            errors.append(sha_error)
        elif actual_sha256 != expected_sha256:
            errors.append(f"{dotted_sha256} must match the seed source review artifact SHA-256")

    return errors


def _review_artifact_preflight(
    raw_artifacts: Any,
    *,
    source_ref: str,
    capability: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Validate that an approved import preserves generated review handoff artifacts."""
    expected_templates = _review_file_lookup(report, "review_evidence_templates")
    expected_packets = _review_file_lookup(report, "review_packets")
    expected_template = expected_templates.get((source_ref, capability), {})
    expected_packet = expected_packets.get((source_ref, capability), {})
    checked = bool(expected_template or expected_packet)
    errors: list[str] = []
    review_artifacts = raw_artifacts if isinstance(raw_artifacts, dict) else {}
    if checked and not isinstance(raw_artifacts, dict):
        errors.append(
            "review_artifacts is required when include_in_training=true and seed source review has generated artifacts"
        )
    errors.extend(
        _review_artifact_field_errors(
            review_artifacts,
            expected_packet,
            path_field="review_packet_path",
            sha256_field="review_packet_sha256",
        )
    )
    errors.extend(
        _review_artifact_field_errors(
            review_artifacts,
            expected_template,
            path_field="review_evidence_template_path",
            sha256_field="review_evidence_template_sha256",
        )
    )
    return {
        "checked": checked,
        "review_packet_path": review_artifacts.get("review_packet_path") or "",
        "review_evidence_template_path": review_artifacts.get("review_evidence_template_path") or "",
        "errors": errors,
    }


def _yolo_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {
            index: str(value)
            for index, value in enumerate(raw_names)
        }
    if isinstance(raw_names, dict):
        names: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return names
    return {}


def _split_for_member(member: str) -> str | None:
    normalized = member.replace("\\", "/").lower()
    for split in ("train", "valid", "val", "test"):
        marker = f"/{split}/"
        if normalized.startswith(f"{split}/") or marker in normalized:
            return "valid" if split == "val" else split
    return None


def _split_payload_key(member: str, *, split: str, folder: str) -> str | None:
    normalized = member.replace("\\", "/").lower()
    aliases = ["val"] if split == "valid" else []
    aliases.insert(0, split)
    for alias in aliases:
        marker = f"/{alias}/{folder}/"
        prefix = f"{alias}/{folder}/"
        if normalized.startswith(prefix):
            rest = normalized[len(prefix):]
        elif marker in normalized:
            rest = normalized.split(marker, 1)[1]
        else:
            continue
        return str(Path(rest).with_suffix(""))
    return None


def _validate_yolo_export_archive(
    path_value: Any,
    *,
    expected_sha256: Any,
    class_mapping: dict[str, Any],
    capability: str,
    expected_counts: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "checked": True,
        "local_path": str(path_value or ""),
        "sha256": "",
        "data_yaml_path": "",
        "source_classes": [],
        "mapped_source_classes": sorted(str(key) for key in class_mapping),
        "mapped_local_classes": sorted(_normalized_mapping_values(class_mapping)),
        "image_count_by_split": {},
        "label_file_count_by_split": {},
        "orphan_label_count_by_split": {},
        "label_file_count_by_local_class": {},
    }
    errors: list[str] = []
    export_path, path_errors = _local_artifact_path(path_value)
    errors.extend(path_errors)
    if export_path is None or path_errors:
        return summary, errors

    actual_sha256 = _sha256_file(export_path)
    summary["local_path"] = _rel(export_path)
    summary["sha256"] = actual_sha256
    expected_sha = str(expected_sha256 or "").strip()
    if expected_sha and len(expected_sha) == 64 and actual_sha256 != expected_sha:
        errors.append("raw_export_local_path SHA-256 does not match raw_export_sha256")
    if not zipfile.is_zipfile(export_path):
        errors.append("raw_export_local_path must be a ZIP archive containing a YOLO export")
        return summary, errors

    try:
        with zipfile.ZipFile(export_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            unsafe_members = [
                name
                for name in members
                if Path(name).is_absolute() or ".." in Path(name).parts
            ]
            if unsafe_members:
                errors.append("raw_export_local_path contains unsafe archive member paths")
                return summary, errors
            data_yaml_candidates = [
                name
                for name in members
                if Path(name).name in {"data.yaml", "dataset.yaml"}
            ]
            if not data_yaml_candidates:
                errors.append("YOLO export archive must include data.yaml or dataset.yaml")
                return summary, errors
            data_yaml_path = sorted(data_yaml_candidates)[0]
            summary["data_yaml_path"] = data_yaml_path
            try:
                data_yaml = yaml.safe_load(archive.read(data_yaml_path).decode("utf-8")) or {}
            except Exception as exc:
                errors.append(f"YOLO export data.yaml is not parseable: {exc}")
                return summary, errors
            if not isinstance(data_yaml, dict):
                errors.append("YOLO export data.yaml must contain a mapping document")
                return summary, errors
            names_by_id = _yolo_names(data_yaml.get("names"))
            if not names_by_id:
                errors.append("YOLO export data.yaml must define class names")
                return summary, errors
            source_classes = set(names_by_id.values())
            summary["source_classes"] = sorted(source_classes)
            missing_source_labels = sorted(
                str(label) for label in class_mapping if str(label) not in source_classes
            )
            if missing_source_labels:
                errors.append(
                    "class_mapping source labels are missing from YOLO export names: "
                    + ", ".join(missing_source_labels)
                )

            image_count_by_split: dict[str, int] = {}
            label_files_by_split: dict[str, int] = {}
            image_keys_by_split: dict[str, set[str]] = {}
            label_keys_by_split: dict[str, set[str]] = {}
            local_class_label_files: dict[str, set[str]] = {}
            for name in members:
                normalized = name.replace("\\", "/")
                split = _split_for_member(normalized)
                suffix = Path(normalized).suffix.lower()
                if split and "/images/" in normalized.lower() and suffix in IMAGE_EXTENSIONS:
                    image_count_by_split[split] = image_count_by_split.get(split, 0) + 1
                    image_key = _split_payload_key(normalized, split=split, folder="images")
                    if image_key:
                        image_keys_by_split.setdefault(split, set()).add(image_key)
                if not (split and "/labels/" in normalized.lower() and suffix == ".txt"):
                    continue
                label_files_by_split[split] = label_files_by_split.get(split, 0) + 1
                label_key = _split_payload_key(normalized, split=split, folder="labels")
                if label_key:
                    label_keys_by_split.setdefault(split, set()).add(label_key)
                try:
                    label_text = archive.read(name).decode("utf-8")
                except Exception as exc:
                    errors.append(f"YOLO label file is not readable: {name}: {exc}")
                    continue
                file_local_classes: set[str] = set()
                for line in label_text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    raw_class_id = stripped.split(maxsplit=1)[0]
                    try:
                        class_id = int(raw_class_id)
                    except ValueError:
                        errors.append(f"YOLO label file has non-integer class id: {name}")
                        continue
                    source_label = names_by_id.get(class_id)
                    if source_label is None:
                        errors.append(f"YOLO label file references unknown class id {class_id}: {name}")
                        continue
                    local_class = str(class_mapping.get(source_label) or "").strip()
                    if local_class:
                        file_local_classes.add(
                            local_class.lower().replace("-", "_").replace(" ", "_")
                        )
                for local_class in file_local_classes:
                    local_class_label_files.setdefault(local_class, set()).add(name)
            summary["image_count_by_split"] = dict(sorted(image_count_by_split.items()))
            summary["label_file_count_by_split"] = dict(sorted(label_files_by_split.items()))
            summary["label_file_count_by_local_class"] = {
                local_class: len(files)
                for local_class, files in sorted(local_class_label_files.items())
            }
            orphan_label_count_by_split: dict[str, int] = {}
            required_classes = REQUIRED_IMPORT_COUNT_CLASSES.get(capability, set())
            for class_name in sorted(required_classes):
                observed = len(local_class_label_files.get(class_name, set()))
                if observed <= 0:
                    errors.append(f"YOLO export has no label files for required local class: {class_name}")
                expected = _count_value(expected_counts, class_name)
                if expected > observed:
                    errors.append(
                        f"YOLO export label-file count for {class_name} is {observed}, "
                        f"below expected_labeled_images_per_class.{class_name}={expected}"
                    )
            for split in ("train", "valid", "test"):
                if image_count_by_split.get(split, 0) <= 0:
                    errors.append(f"YOLO export archive must include {split} images")
                if label_files_by_split.get(split, 0) <= 0:
                    errors.append(f"YOLO export archive must include {split} labels")
                orphan_labels = sorted(
                    label_keys_by_split.get(split, set()) - image_keys_by_split.get(split, set())
                )
                if orphan_labels:
                    orphan_label_count_by_split[split] = len(orphan_labels)
                if orphan_labels:
                    sample = ", ".join(orphan_labels[:5])
                    if len(orphan_labels) > 5:
                        sample += f", +{len(orphan_labels) - 5} more"
                    errors.append(
                        f"YOLO export {split} labels must have matching images: {sample}"
                    )
            summary["orphan_label_count_by_split"] = dict(sorted(orphan_label_count_by_split.items()))
    except zipfile.BadZipFile:
        errors.append("raw_export_local_path must be a readable ZIP archive")
    return summary, errors


def _review_evidence_errors(
    path_value: Any,
    sha256_value: Any,
    *,
    source_ref: str = "",
    capability: str = "",
    source_url: str = "",
    license_note: str = "",
    reviewed_by: str = "",
    reviewed_at: str = "",
) -> list[str]:
    errors: list[str] = []
    raw_path = str(path_value or "").strip()
    raw_sha256 = str(sha256_value or "").strip()
    evidence_path: Path | None = None
    if not raw_path:
        errors.append("review_evidence_path is required")
    elif "://" in raw_path:
        errors.append("review_evidence_path must be a local evidence file path, not a URL")
    else:
        evidence_path = Path(raw_path).expanduser()
        if not evidence_path.is_absolute():
            evidence_path = ROOT / evidence_path
        if not evidence_path.is_file():
            errors.append("review_evidence_path must point to an existing local file")

    sha_error = _sha256_error(raw_sha256, "review_evidence_sha256")
    if sha_error:
        errors.append(sha_error)

    if evidence_path is not None and evidence_path.is_file() and not sha_error:
        actual_sha256 = _sha256_file(evidence_path)
        if actual_sha256 != raw_sha256:
            errors.append("review_evidence_sha256 does not match review_evidence_path")
        try:
            evidence_doc = yaml.safe_load(evidence_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            errors.append(f"review_evidence_path is not parseable YAML/JSON: {exc}")
            evidence_doc = {}
        if not isinstance(evidence_doc, dict):
            errors.append("review_evidence_path must contain a mapping document")
            evidence_doc = {}
        if evidence_doc:
            if evidence_doc.get("kind") != REVIEW_EVIDENCE_KIND:
                errors.append(f"review evidence kind must be {REVIEW_EVIDENCE_KIND}")
            if evidence_doc.get("version") != 1:
                errors.append("review evidence version must be 1")
            if source_ref and evidence_doc.get("source_ref") != source_ref:
                errors.append("review evidence source_ref must match the approved source")
            if capability and evidence_doc.get("capability") != capability:
                errors.append("review evidence capability must match the approved source")
            if source_url and evidence_doc.get("source_url") != source_url:
                errors.append("review evidence source_url must match the approved source")
            if license_note and evidence_doc.get("license_note") != license_note:
                errors.append("review evidence license_note must match the approved source")
            if reviewed_by and evidence_doc.get("reviewed_by") != reviewed_by:
                errors.append("review evidence reviewed_by must match the approved source")
            if reviewed_at and evidence_doc.get("reviewed_at") != reviewed_at:
                errors.append("review evidence reviewed_at must match the approved source")
            review_items = (
                evidence_doc.get("review_items")
                if isinstance(evidence_doc.get("review_items"), dict)
                else {}
            )
            if not review_items:
                errors.append("review evidence review_items are required")
            for item in sorted(REQUIRED_REVIEW_ITEMS):
                review_item = review_items.get(item) if isinstance(review_items.get(item), dict) else {}
                if review_item.get("approved") is not True:
                    errors.append(f"review evidence {item}.approved must be true")
                if review_item.get("prefilled_by_agent") is True:
                    errors.append(
                        f"review evidence {item}.prefilled_by_agent must be false or removed"
                    )
                if not str(review_item.get("evidence_ref") or "").strip():
                    errors.append(f"review evidence {item}.evidence_ref is required")
    return errors


def _candidate_report(
    candidate: dict[str, Any],
    source: dict[str, Any] | None,
    *,
    source_research_max_age_days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    source_ref = str(candidate.get("source_ref") or "")
    capability = str(candidate.get("capability") or "")
    approval_status = str(candidate.get("approval_status") or "")
    approved_for_training = bool(candidate.get("approved_for_training"))
    required_review = _as_string_set(candidate.get("required_review"))
    completed_review = (
        candidate.get("completed_review") if isinstance(candidate.get("completed_review"), dict) else {}
    )
    review_priority = _review_priority(candidate.get("review_priority"))
    review_focus = _normalize_review_focus(candidate.get("review_focus"))
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []

    missing_candidate_fields = sorted(field for field in REQUIRED_CANDIDATE_FIELDS if field not in candidate)
    if missing_candidate_fields:
        errors.append(f"missing candidate fields: {', '.join(missing_candidate_fields)}")
    if capability not in ALLOWED_CAPABILITIES:
        errors.append(f"invalid capability: {capability or 'missing'}")
    if str(candidate.get("status") or "") != "candidate_seed_only":
        errors.append(f"candidate status must be candidate_seed_only: {candidate.get('status')}")
    if review_priority is None:
        errors.append("review_priority must be a positive integer")
    if not review_focus:
        errors.append("review_focus is required")
    if approval_status not in ALLOWED_APPROVAL_STATUS:
        errors.append(f"invalid approval_status: {approval_status or 'missing'}")
    if source is None:
        errors.append(f"source_ref does not exist in shared_sources: {source_ref or 'missing'}")
        source = {}
    legal_references = _legal_references_for_source(source)
    source_for_research = dict(source)
    if not isinstance(source_for_research.get("legal_references"), dict) or not source_for_research.get("legal_references"):
        source_for_research["legal_references"] = legal_references
    source_research_status = _source_research_status(source_for_research)
    source_research_disposition = str(source.get("source_research_disposition") or "open").strip()
    if source_research_disposition not in ALLOWED_SOURCE_RESEARCH_DISPOSITIONS:
        errors.append(
            "invalid source_research_disposition: "
            f"{source_research_disposition or 'missing'}"
        )

    missing_source_fields = sorted(field for field in REQUIRED_SHARED_SOURCE_FIELDS if not source.get(field))
    if missing_source_fields:
        errors.append(f"shared source missing fields: {', '.join(missing_source_fields)}")
    source_facts = _source_facts(source.get("source_facts"))
    if review_priority is not None and review_priority <= SOURCE_FACT_REVIEW_PRIORITY_MAX:
        missing_source_fact_fields = sorted(
            field for field in REQUIRED_SOURCE_FACT_FIELDS if not source_facts.get(field)
        )
        if missing_source_fact_fields:
            warnings.append(
                "high-priority source missing source_facts fields: "
                + ", ".join(missing_source_fact_fields)
            )
        if not any(source_facts.get(field) for field in SOURCE_FACT_COUNT_FIELDS):
            warnings.append(
                "high-priority source missing source_facts image-count field: "
                + " or ".join(sorted(SOURCE_FACT_COUNT_FIELDS))
            )
    checked_value = source.get("checked")
    checked_date = _parse_checked_date(checked_value)
    if checked_value and checked_date is None:
        errors.append("shared source checked must be YYYY-MM-DD or ISO datetime")
    elif checked_date is not None:
        age_days = ((now or datetime.now(timezone.utc)).date() - checked_date).days
        if age_days < 0:
            errors.append(f"shared source checked is in the future: {checked_date.isoformat()}")
        elif source_research_max_age_days is not None and age_days > source_research_max_age_days:
            errors.append(
                "shared source research is stale: "
                f"checked={checked_date.isoformat()} age_days={age_days} "
                f"max_age_days={source_research_max_age_days}"
            )

    missing_review_items = sorted(REQUIRED_REVIEW_ITEMS - required_review)
    if missing_review_items:
        errors.append(f"required_review missing items: {', '.join(missing_review_items)}")

    blocker_text = str(candidate.get("blocker") or "").strip()
    if not approved_for_training:
        blockers.append(f"{source_ref}: source review not approved for training")
    if approval_status in {"unreviewed", "rejected"}:
        blockers.append(f"{source_ref}: approval_status={approval_status}")
    if not approved_for_training and not blocker_text:
        errors.append("candidate blocker is required until approved_for_training=true")
    if approved_for_training and blocker_text:
        errors.append("approved training source must clear blocker")
    if approval_status == "approved_for_training" and not approved_for_training:
        errors.append("approval_status=approved_for_training requires approved_for_training=true")
    if approved_for_training and approval_status != "approved_for_training":
        errors.append("approved_for_training=true requires approval_status=approved_for_training")
    if approved_for_training and not candidate.get("manifest_import_path"):
        errors.append("approved training source must include manifest_import_path")
    if approved_for_training and not candidate.get("reviewed_by"):
        errors.append("approved training source must include reviewed_by")
    if approved_for_training and not candidate.get("reviewed_at"):
        errors.append("approved training source must include reviewed_at")
    if approved_for_training and source_research_status.get("evidence_ready_for_human_review") is not True:
        missing_research = source_research_status.get("missing") or []
        suffix = ": " + ", ".join(str(item) for item in missing_research) if missing_research else ""
        errors.append("approved training source requires evidence-ready source research" + suffix)
    if approved_for_training:
        errors.extend(
            "approved training source " + error
            for error in _review_evidence_errors(
                candidate.get("review_evidence_path"),
                candidate.get("review_evidence_sha256"),
                source_ref=source_ref,
                capability=capability,
                source_url=str(source.get("url") or ""),
                license_note=str(source.get("license_note") or ""),
                reviewed_by=str(candidate.get("reviewed_by") or ""),
                reviewed_at=str(candidate.get("reviewed_at") or ""),
            )
        )
    if approved_for_training:
        missing_completed_review = sorted(
            item for item in REVIEW_BOOLEAN_FIELDS if completed_review.get(item) is not True
        )
        if missing_completed_review:
            errors.append(
                "approved training source completed_review missing approvals: "
                + ", ".join(missing_completed_review)
            )

    decision = str(source.get("decision") or "")
    license_note = str(source.get("license_note") or "")
    if (
        "CC BY 4.0" not in license_note
        and "Apache-2.0" not in license_note
        and "Public Domain" not in license_note
    ):
        warnings.append("license_note is not clearly permissive; keep blocked until legal review")
    if "candidate" not in decision and "seed" not in decision:
        warnings.append("source decision does not clearly state seed/candidate status")

    return {
        "source_ref": source_ref,
        "capability": capability,
        "url": source.get("url"),
        "checked": source.get("checked"),
        "license_note": source.get("license_note"),
        "legal_references": legal_references,
        "source_facts": source_facts,
        "source_research_status": source_research_status,
        "source_research_disposition": source_research_disposition,
        "source_research_evidence": source.get("source_research_evidence")
        if isinstance(source.get("source_research_evidence"), list)
        else [],
        "suggested_mapping": _suggested_source_mapping(capability, source_facts),
        "relevance": source.get("relevance"),
        "decision": source.get("decision"),
        "status": candidate.get("status"),
        "review_priority": review_priority,
        "review_focus": review_focus,
        "approval_status": approval_status,
        "approved_for_training": approved_for_training,
        "manifest_import_path": candidate.get("manifest_import_path"),
        "reviewed_by": candidate.get("reviewed_by"),
        "reviewed_at": candidate.get("reviewed_at"),
        "review_evidence_path": candidate.get("review_evidence_path"),
        "review_evidence_sha256": candidate.get("review_evidence_sha256"),
        "blocker": candidate.get("blocker"),
        "required_review": sorted(required_review),
        "completed_review": completed_review,
        "errors": errors,
        "warnings": warnings,
        "blockers": blockers,
        "training_usable": approved_for_training and not errors,
    }


def _audit_seed_sources_doc(
    doc: dict[str, Any],
    model_packs_path: Path = DEFAULT_MODEL_PACKS,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    policy = doc.get("policy") if isinstance(doc.get("policy"), dict) else {}
    source_research_max_age_days = None
    if "source_research_max_age_days" in policy:
        source_research_max_age_days = _positive_int(policy.get("source_research_max_age_days"))
        if source_research_max_age_days is None:
            errors.append("policy.source_research_max_age_days must be a positive integer")
    shared_sources = doc.get("shared_sources") if isinstance(doc.get("shared_sources"), dict) else {}
    source_of_truth = doc.get("source_of_truth") if isinstance(doc.get("source_of_truth"), dict) else {}
    factory_pack = ((doc.get("packs") or {}).get("factory_ppe_3cam") or {})
    shared_sources = doc.get("shared_sources") if isinstance(doc.get("shared_sources"), dict) else {}
    sourcing = factory_pack.get("sourcing_status") if isinstance(factory_pack.get("sourcing_status"), dict) else {}
    raw_candidates = sourcing.get("candidate_sources") if isinstance(sourcing.get("candidate_sources"), list) else []

    if not raw_candidates:
        errors.append("factory_ppe_3cam.sourcing_status.candidate_sources must not be empty")

    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            errors.append("candidate_sources entries must be objects")
            continue
        source_ref = str(raw.get("source_ref") or "")
        source = shared_sources.get(source_ref) if isinstance(shared_sources.get(source_ref), dict) else None
        candidates.append(
            _candidate_report(
                raw,
                source,
                source_research_max_age_days=source_research_max_age_days,
                now=now,
            )
        )

    candidate_errors = [error for item in candidates for error in item["errors"]]
    candidate_warnings = [warning for item in candidates for warning in item["warnings"]]
    candidate_blockers = [blocker for item in candidates for blocker in item["blockers"]]
    errors.extend(candidate_errors)
    warnings.extend(candidate_warnings)

    capability_coverage = sorted({item["capability"] for item in candidates if item.get("capability")})
    missing_capabilities = sorted(ALLOWED_CAPABILITIES - set(capability_coverage))
    if missing_capabilities:
        errors.append(f"seed source candidates missing capability coverage: {', '.join(missing_capabilities)}")

    training_usable = [item for item in candidates if item.get("training_usable")]
    training_usable_capabilities = sorted({item["capability"] for item in training_usable})
    missing_training_capabilities = sorted(ALLOWED_CAPABILITIES - set(training_usable_capabilities))
    blockers = list(candidate_blockers)
    if missing_training_capabilities:
        blockers.append(
            "approved training seed sources missing capability coverage: "
            + ", ".join(missing_training_capabilities)
        )

    gate_passed = not errors and not blockers and not missing_training_capabilities
    report = {
        "ok": not errors,
        "generated_at": utc_now(),
        "inputs": {
            "model_packs": _rel(model_packs_path),
            "source_research_max_age_days": source_research_max_age_days,
        },
        "gate_passed": gate_passed,
        "status": "approved_for_manifest_import" if gate_passed else "blocked_pending_source_review",
        "candidate_count": len(candidates),
        "training_usable_count": len(training_usable),
        "capability_coverage": capability_coverage,
        "training_usable_capability_coverage": training_usable_capabilities,
        "missing_training_capabilities": missing_training_capabilities,
        "source_research_readiness": _source_research_readiness_summary(candidates),
        "review_queue_summary": _source_review_queue_summary(candidates),
        "candidates": candidates,
        "blockers": blockers,
        "errors": errors,
        "warnings": warnings,
    }
    source_recheck_path = source_of_truth.get("apron_harness_source_recheck")
    if source_recheck_path:
        resolved_source_recheck = _resolve_local_path(source_recheck_path)
        source_recheck_exists = bool(resolved_source_recheck and resolved_source_recheck.is_file())
        report["source_recheck"] = {
            "path": str(source_recheck_path),
            "exists": source_recheck_exists,
            "sha256": _sha256_file(resolved_source_recheck) if source_recheck_exists else None,
            "evidence_boundary": (
                "Fresh source research evidence only; this does not approve any source "
                "for training or authorize public export import."
            ),
        }
    report["review_queue"] = build_review_queue(report)
    return report


def audit_seed_sources(model_packs_path: Path = DEFAULT_MODEL_PACKS) -> dict[str, Any]:
    return _audit_seed_sources_doc(_load_yaml(model_packs_path), model_packs_path)


def build_import_manifest_template(
    report: dict[str, Any],
    seed_review_report: str | None = None,
    source_review_sha256: str | None = None,
    source_refs: set[str] | None = None,
    template_scope: str = "all_reviewable_sources",
) -> dict[str, Any]:
    """Return a fillable seed-import manifest for reviewed public seed sources."""
    imports: list[dict[str, Any]] = []
    evidence_templates = _review_file_lookup(report, "review_evidence_templates")
    review_packets = _review_file_lookup(report, "review_packets")
    for candidate in sorted(report.get("candidates") or [], key=_candidate_sort_key):
        if not _is_reviewable_source_candidate(candidate):
            continue
        source_ref = str(candidate.get("source_ref") or "")
        if source_refs is not None and source_ref not in source_refs:
            continue
        capability = str(candidate.get("capability") or "")
        evidence_template = evidence_templates.get((source_ref, capability), {})
        review_packet = review_packets.get((source_ref, capability), {})
        imports.append(
            {
                "review_priority": candidate.get("review_priority"),
                "review_focus": candidate.get("review_focus"),
                "source_ref": source_ref,
                "capability": capability,
                "source_url": candidate.get("url"),
                "license_note": candidate.get("license_note"),
                "review_artifacts": {
                    "review_packet_path": review_packet.get("path") or "",
                    "review_packet_sha256": review_packet.get("sha256") or "",
                    "review_evidence_template_path": evidence_template.get("path") or "",
                    "review_evidence_template_sha256": evidence_template.get("sha256") or "",
                },
                "approval_preconditions": [
                    "source review evidence YAML is filled by a human/legal reviewer",
                    "review evidence SHA-256 is copied into the review checklist",
                    "review checklist is applied to a reviewed model_packs.yaml",
                    "source has approval_status=approved_for_training and approved_for_training=true",
                    "seed import manifest validates the reviewed YOLO export archive before include_in_training=true",
                ],
                "include_in_training": False,
                "review_status": "needs_review",
                "reviewed_by": "",
                "reviewed_at": "",
                "manifest_import_path": "",
                "raw_export_ref": "",
                "raw_export_sha256": "",
                "raw_export_local_path": "",
                "export_format": "yolo",
                "completed_review": {
                    item: False for item in sorted(REVIEW_BOOLEAN_FIELDS)
                },
                "suggested_mapping": candidate.get("suggested_mapping") or {},
                "seed_import_fill_plan": _seed_import_fill_plan(candidate),
                "class_mapping": {},
                "person_box_policy": "",
                "hard_negative_policy": "",
                "split_plan": {
                    "train": "",
                    "val": "",
                    "test": "",
                },
                "expected_labeled_images_per_class": {
                    "person": 0,
                    "apron": 0,
                    "safety_harness": 0,
                    "safety_lanyard": 0,
                },
                "notes": (
                    "Fill this only after legal/export/provenance review. Keep include_in_training=false until "
                    "approved. Included imports must record a remote immutable raw_export_ref, raw_export_sha256, "
                    "a local reviewed YOLO export archive at raw_export_local_path with the same SHA-256, YOLO "
                    "export format, image/label pairing, person-box policy, hard-negative policy, class mapping, "
                    "split plan, and nonzero expected counts for person plus the target PPE class."
                ),
            }
        )
    return {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "generated_at": report.get("generated_at"),
        "source_review_report": seed_review_report or "",
        "source_review_sha256": source_review_sha256 or source_review_fingerprint(report),
        "source_review_inputs": report.get("inputs", {}),
        "template_scope": template_scope,
        "selected_source_refs": sorted(source_refs) if source_refs is not None else [],
        "fill_contract": {
            "approval_boundary": (
                "This template is not approval. Keep every import include_in_training=false "
                "until the matching review evidence YAML, checklist row, reviewed model_packs.yaml, "
                "and YOLO export preflight all pass."
            ),
            "required_before_include_in_training": [
                "approval_status=approved_for_training in reviewed model_packs.yaml",
                "approved_for_training=true in reviewed model_packs.yaml",
                "review_status=approved_for_training",
                "reviewed_by and reviewed_at are filled",
                "manifest_import_path points to the approved import manifest row source",
                "raw_export_ref is a remote immutable export reference",
                "raw_export_sha256 is the SHA-256 of the reviewed YOLO export ZIP",
                "raw_export_local_path points to the reviewed local YOLO export ZIP",
                "class_mapping maps required local classes to reviewed source labels",
                "person_box_policy and hard_negative_policy are documented",
                "split_plan.train, split_plan.val, and split_plan.test are documented",
                "expected_labeled_images_per_class has nonzero counts for person plus target PPE classes",
            ],
            "forbidden_until_approved": [
                "include_in_training=true",
                "training_usable=true",
                "approval_status=approved_for_training",
                "approved_for_training=true",
            ],
            "validation_commands": [
                (
                    ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
                    "--model-packs /path/to/reviewed/model_packs.yaml "
                    "--validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml"
                ),
                (
                    ".venv/bin/python scripts/apron_harness_readiness_doctor.py "
                    "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
                    "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml "
                    "--out qa/video_eval/results/apron_harness_readiness_doctor.json"
                ),
            ],
        },
        "imports": imports,
    }


def minimum_approval_source_refs(report: dict[str, Any]) -> set[str]:
    """Return source refs from the non-approving priority coverage plan."""
    coverage_plan = build_source_coverage_plan(report)
    refs: set[str] = set()
    capabilities = coverage_plan.get("capabilities") if isinstance(coverage_plan, dict) else {}
    if not isinstance(capabilities, dict):
        return refs
    for capability in ("apron_required", "harness_required"):
        capability_plan = capabilities.get(capability)
        if not isinstance(capability_plan, dict):
            continue
        priority_plan = capability_plan.get("priority_coverage_plan")
        if not isinstance(priority_plan, dict):
            continue
        for source in priority_plan.get("selected_sources") or []:
            if isinstance(source, dict) and source.get("source_ref"):
                refs.add(str(source["source_ref"]))
    return refs


def build_review_checklist_csv(report: dict[str, Any]) -> str:
    """Return a fillable CSV for source-by-source public seed review."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REVIEW_CHECKLIST_FIELDS, lineterminator="\n")
    writer.writeheader()
    evidence_templates = _review_file_lookup(report, "review_evidence_templates")
    review_packets = _review_file_lookup(report, "review_packets")
    import_template = report.get("seed_import_manifest_template")
    if not isinstance(import_template, dict):
        import_template = {}
    for candidate in sorted(report.get("candidates") or [], key=_candidate_sort_key):
        if not _is_reviewable_source_candidate(candidate):
            continue
        source_ref = str(candidate.get("source_ref") or "")
        capability = str(candidate.get("capability") or "")
        evidence_template = evidence_templates.get((source_ref, capability), {})
        review_packet = review_packets.get((source_ref, capability), {})
        row = {
            "review_priority": candidate.get("review_priority"),
            "source_ref": source_ref,
            "capability": capability,
            "source_url": candidate.get("url"),
            "license_note": candidate.get("license_note"),
            "approval_status": candidate.get("approval_status") or "unreviewed",
            "approved_for_training": str(bool(candidate.get("approved_for_training"))).lower(),
            "training_usable": str(bool(candidate.get("training_usable"))).lower(),
            "current_blocker": candidate.get("blocker") or "",
            "review_focus": candidate.get("review_focus") or "",
            "reviewed_by": candidate.get("reviewed_by") or "",
            "reviewed_at": candidate.get("reviewed_at") or "",
            "manifest_import_path": candidate.get("manifest_import_path") or "",
            "review_packet_path": review_packet.get("path") or "",
            "review_packet_sha256": review_packet.get("sha256") or "",
            "review_evidence_template_path": evidence_template.get("path") or "",
            "review_evidence_template_sha256": evidence_template.get("sha256") or "",
            "seed_import_manifest_template_path": import_template.get("path") or "",
            "seed_import_manifest_template_sha256": import_template.get("sha256") or "",
            "review_evidence_path": candidate.get("review_evidence_path") or "",
            "review_evidence_sha256": candidate.get("review_evidence_sha256") or "",
            "review_notes": "",
        }
        for field in sorted(REVIEW_BOOLEAN_FIELDS):
            row[field] = "false"
        writer.writerow(row)
    return output.getvalue()


def _append_unique(values: list[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in values:
        values.append(text)


def _matching_research_refs(candidate: dict[str, Any], terms: tuple[str, ...], limit: int = 4) -> list[str]:
    refs: list[str] = []
    evidence = candidate.get("source_research_evidence")
    if not isinstance(evidence, list):
        return refs
    for item in evidence:
        if not isinstance(item, dict):
            continue
        haystack = " ".join(
            str(item.get(field) or "")
            for field in ("finding", "evidence_ref", "implication")
        ).lower()
        if any(term in haystack for term in terms):
            _append_unique(refs, item.get("evidence_ref"))
        if len(refs) >= limit:
            break
    return refs


def _review_hint(
    *,
    refs: list[str],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "evidence_refs": refs,
        "notes": " ".join(note for note in notes if note),
    }


def build_agent_collected_review_hints(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return non-approving evidence hints grouped by required review item."""
    legal_references = (
        candidate.get("legal_references")
        if isinstance(candidate.get("legal_references"), dict)
        else {}
    )
    platform_references = (
        legal_references.get("platform_references")
        if isinstance(legal_references.get("platform_references"), dict)
        else {}
    )
    source_facts = _source_facts(candidate.get("source_facts"))
    suggested_mapping = (
        candidate.get("suggested_mapping")
        if isinstance(candidate.get("suggested_mapping"), dict)
        else {}
    )
    local_mapping = (
        suggested_mapping.get("local_class_to_source_labels")
        if isinstance(suggested_mapping.get("local_class_to_source_labels"), dict)
        else {}
    )
    hard_negative_labels = suggested_mapping.get("hard_negative_source_labels")
    if not isinstance(hard_negative_labels, list):
        hard_negative_labels = []
    mapping_warnings = suggested_mapping.get("warnings")
    if not isinstance(mapping_warnings, list):
        mapping_warnings = []

    common_source_refs: list[str] = []
    _append_unique(common_source_refs, candidate.get("url"))
    _append_unique(common_source_refs, legal_references.get("source_url"))

    license_refs = list(common_source_refs)
    _append_unique(license_refs, legal_references.get("license_reference_url"))
    _append_unique(license_refs, platform_references.get("platform_terms_url"))
    _append_unique(license_refs, platform_references.get("platform_licensing_url"))
    for ref in _matching_research_refs(
        candidate,
        ("license", "cc by", "public domain", "terms", "commercial", "attribution"),
    ):
        _append_unique(license_refs, ref)

    export_refs = list(common_source_refs)
    _append_unique(export_refs, platform_references.get("export_docs_url"))
    _append_unique(export_refs, platform_references.get("universe_download_docs_url"))
    for ref in _matching_research_refs(
        candidate,
        ("export", "download", "yolo", "private account key", "terms"),
    ):
        _append_unique(export_refs, ref)

    provenance_refs = list(common_source_refs)
    for ref in _matching_research_refs(
        candidate,
        ("dataset", "version", "author", "cite this project", "image", "source page"),
    ):
        _append_unique(provenance_refs, ref)

    privacy_refs = list(common_source_refs)
    _append_unique(privacy_refs, legal_references.get("license_reference_url"))
    for ref in _matching_research_refs(
        candidate,
        ("privacy", "publicity", "moral", "identifiable", "people", "private"),
    ):
        _append_unique(privacy_refs, ref)

    class_mapping_refs = list(common_source_refs)
    for ref in _matching_research_refs(
        candidate,
        ("class", "label", "person", "apron", "harness", "lanyard", "sample"),
    ):
        _append_unique(class_mapping_refs, ref)

    person_box_refs = list(common_source_refs)
    for ref in _matching_research_refs(
        candidate,
        ("person", "human", "worker", "body", "sample label", "label"),
    ):
        _append_unique(person_box_refs, ref)

    hard_negative_refs = list(common_source_refs)
    for ref in _matching_research_refs(
        candidate,
        ("hard-negative", "hard negative", "negative", "no_apron", "no harness", "no-safety", "noisy", "vest", "ladder", "scaffold"),
    ):
        _append_unique(hard_negative_refs, ref)

    split_refs = list(common_source_refs)
    for ref in _matching_research_refs(
        candidate,
        ("train", "valid", "test", "split", "augmentation", "resize"),
    ):
        _append_unique(split_refs, ref)

    manifest_refs = list(export_refs)
    for ref in class_mapping_refs:
        _append_unique(manifest_refs, ref)

    dataset_version = source_facts.get("dataset_version")
    split = dataset_version.get("split") if isinstance(dataset_version, dict) else None
    return {
        "license_terms": _review_hint(
            refs=license_refs,
            notes=[
                "Agent prefill only; reviewer must confirm commercial training rights, attribution, and platform terms.",
                f"License note observed: {candidate.get('license_note') or 'missing'}.",
            ],
        ),
        "export_terms": _review_hint(
            refs=export_refs,
            notes=[
                "Agent prefill only; reviewer must confirm export is allowed for commercial local training and record immutable export storage.",
                "If Roboflow export code is used, private account keys must not be committed.",
            ],
        ),
        "dataset_card_provenance": _review_hint(
            refs=provenance_refs,
            notes=[
                "Agent prefill only; reviewer must archive the source page, citation/license block, author, dataset/version ID, and access date.",
                f"Observed author: {source_facts.get('source_author') or 'missing'}.",
            ],
        ),
        "privacy_and_identity_risk": _review_hint(
            refs=privacy_refs,
            notes=[
                "Agent prefill only; reviewer must assess identifiable people, sensitive locations, minors, redaction/consent, and non-copyright rights.",
            ],
        ),
        "class_mapping": _review_hint(
            refs=class_mapping_refs,
            notes=[
                "Agent prefill only; reviewer must confirm or replace the suggested class mapping before import.",
                f"Suggested local mapping keys: {', '.join(sorted(local_mapping)) or 'none'}.",
            ],
        ),
        "person_box_coverage": _review_hint(
            refs=person_box_refs,
            notes=[
                "Agent prefill only; reviewer must verify every training frame has usable person boxes or a documented policy to add/review them.",
                f"Observed classes include: {', '.join(str(item) for item in source_facts.get('classes') or []) or 'none'}.",
            ],
        ),
        "hard_negative_coverage": _review_hint(
            refs=hard_negative_refs,
            notes=[
                "Agent prefill only; reviewer must decide whether negative/confuser labels are adequate for production data.",
                f"Suggested hard negatives: {', '.join(str(item) for item in hard_negative_labels) or 'none'}.",
                f"Mapping warnings: {'; '.join(str(item) for item in mapping_warnings) or 'none'}.",
            ],
        ),
        "train_val_test_split": _review_hint(
            refs=split_refs,
            notes=[
                "Agent prefill only; reviewer must confirm leakage controls, source-version boundaries, and final train/val/test counts.",
                f"Observed source split: {split if isinstance(split, dict) else 'missing'}.",
            ],
        ),
        "manifest_import_plan": _review_hint(
            refs=manifest_refs,
            notes=[
                "Agent prefill only; reviewer must fill raw_export_ref, raw_export_sha256, raw_export_local_path, split plan, class mapping, and expected class counts.",
                "Do not set include_in_training=true until the checklist and reviewed export validation pass.",
            ],
        ),
    }


def _review_item_template(item: str, hint: dict[str, Any]) -> dict[str, Any]:
    refs = hint.get("evidence_refs") if isinstance(hint.get("evidence_refs"), list) else []
    notes = str(hint.get("notes") or "").strip()
    if notes:
        notes = (
            notes
            + " Human reviewer must remove or set prefilled_by_agent=false after verifying this item."
        )
    else:
        notes = "Agent prefill only. Human reviewer must verify this item before approval."
    return {
        "approved": False,
        "prefilled_by_agent": True,
        "evidence_ref": "; ".join(str(ref) for ref in refs if str(ref).strip()),
        "notes": notes,
        "guidance": REVIEW_ITEM_GUIDANCE[item],
    }


def build_review_evidence_template(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a non-approving review evidence bundle template for one candidate source."""
    source_ref = str(candidate.get("source_ref") or "")
    capability = str(candidate.get("capability") or "")
    agent_collected_review_hints = build_agent_collected_review_hints(candidate)
    return {
        "kind": REVIEW_EVIDENCE_KIND,
        "version": 1,
        "source_ref": source_ref,
        "capability": capability,
        "source_url": candidate.get("url") or "",
        "license_note": candidate.get("license_note") or "",
        "legal_references": candidate.get("legal_references") or {},
        "source_facts": _source_facts(candidate.get("source_facts")),
        "source_research_evidence": candidate.get("source_research_evidence") or [],
        "suggested_mapping": candidate.get("suggested_mapping") or {},
        "agent_collected_review_hints": agent_collected_review_hints,
        "review_focus": candidate.get("review_focus") or "",
        "agent_research_boundary": {
            "can_collect_evidence": True,
            "can_approve_training": False,
            "agent_research_tasks": AGENT_RESEARCH_TASKS,
            "human_approval_tasks": HUMAN_APPROVAL_TASKS,
            "notes": (
                "Agent/browser research can gather and cite source facts, but it cannot approve "
                "commercial training use. A human/legal reviewer must set review_items.*.approved=true "
                "and sign reviewed_by/reviewed_at."
            ),
        },
        "reviewed_by": "",
        "reviewed_at": "",
        "review_items": {
            item: _review_item_template(
                item,
                agent_collected_review_hints.get(item) or {},
            )
            for item in sorted(REQUIRED_REVIEW_ITEMS)
        },
        "notes": (
            "Agent-collected review hints are not approval. Fill reviewed_by/reviewed_at, verify every "
            "review_items.* entry, remove or set prefilled_by_agent=false for each approved item, and keep "
            "approved=false until the reviewer accepts the evidence. After filling, record this file path "
            "in review_evidence_path and its SHA-256 in review_evidence_sha256."
        ),
    }


def write_review_evidence_templates(report: dict[str, Any], template_dir: Path) -> list[dict[str, Any]]:
    """Write one review evidence template per seed-source candidate."""
    template_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    reviewable_candidates = [
        candidate
        for candidate in sorted(report.get("candidates") or [], key=_candidate_sort_key)
        if _is_reviewable_source_candidate(candidate)
    ]
    keep_filenames = {
        f"{_slug(candidate.get('source_ref'))}__{_slug(candidate.get('capability'))}.review_evidence.yaml"
        for candidate in reviewable_candidates
    }
    _prune_stale_generated_files(template_dir, "*.review_evidence.yaml", keep_filenames)
    for candidate in reviewable_candidates:
        source_ref = str(candidate.get("source_ref") or "")
        capability = str(candidate.get("capability") or "")
        filename = f"{_slug(source_ref)}__{_slug(capability)}.review_evidence.yaml"
        path = template_dir / filename
        path.write_text(
            yaml.safe_dump(build_review_evidence_template(candidate), sort_keys=False),
            encoding="utf-8",
        )
        written.append(
            {
                "source_ref": source_ref,
                "capability": capability,
                "path": _rel(path),
                "sha256": _sha256_file(path),
            }
        )
    return written


def _review_template_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    template_files = ((report.get("review_evidence_templates") or {}).get("files") or [])
    if not template_files:
        return []
    candidates_by_key = {
        (str(candidate.get("source_ref") or ""), str(candidate.get("capability") or "")): candidate
        for candidate in report.get("candidates") or []
    }
    rows: list[dict[str, Any]] = []
    for item in template_files:
        source_ref = str(item.get("source_ref") or "")
        capability = str(item.get("capability") or "")
        candidate = candidates_by_key.get((source_ref, capability), {})
        rows.append(
            {
                "review_priority": _review_priority(candidate.get("review_priority")) or 999999,
                "source_ref": source_ref,
                "capability": capability,
                "path": item.get("path") or "",
                "sha256": item.get("sha256") or "",
            }
        )
    return sorted(rows, key=lambda row: (row["review_priority"], row["source_ref"], row["capability"]))


def _review_packet_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    packet_files = ((report.get("review_packets") or {}).get("files") or [])
    if not packet_files:
        return []
    candidates_by_key = {
        (str(candidate.get("source_ref") or ""), str(candidate.get("capability") or "")): candidate
        for candidate in report.get("candidates") or []
    }
    rows: list[dict[str, Any]] = []
    for item in packet_files:
        source_ref = str(item.get("source_ref") or "")
        capability = str(item.get("capability") or "")
        candidate = candidates_by_key.get((source_ref, capability), {})
        rows.append(
            {
                "review_priority": _review_priority(candidate.get("review_priority")) or 999999,
                "source_ref": source_ref,
                "capability": capability,
                "path": item.get("path") or "",
                "sha256": item.get("sha256") or "",
            }
        )
    return sorted(rows, key=lambda row: (row["review_priority"], row["source_ref"], row["capability"]))


def _review_file_lookup(report: dict[str, Any], key: str) -> dict[tuple[str, str], dict[str, Any]]:
    artifact_group = report.get(key)
    if not isinstance(artifact_group, dict):
        return {}
    files = artifact_group.get("files") or []
    if not isinstance(files, list):
        return {}
    return {
        (str(item.get("source_ref") or ""), str(item.get("capability") or "")): item
        for item in files
        if isinstance(item, dict)
    }


def build_review_packet(candidate: dict[str, Any], evidence_template: dict[str, Any] | None = None) -> str:
    """Return a human-readable review packet for one public seed-source candidate."""
    priority = _review_priority(candidate.get("review_priority")) or 999999
    source_ref = str(candidate.get("source_ref") or "")
    capability = str(candidate.get("capability") or "")
    required_classes = ", ".join(sorted(REQUIRED_IMPORT_COUNT_CLASSES.get(capability, set()))) or "person plus target PPE"
    lines = [
        f"# {source_ref} / {capability} Review Packet",
        "",
        "This packet is a source-review aid, not approval to train. Keep `approved_for_training=false` until the checklist, evidence bundle, and import manifest all pass the seed-source gates.",
        "",
        "## Source Facts",
        "",
        f"- Priority: `{priority}`",
        f"- Source ref: `{source_ref}`",
        f"- Capability: `{capability}`",
        f"- Source URL: {candidate.get('url') or ''}",
        f"- Checked: `{candidate.get('checked') or ''}`",
        f"- License note: {_md_cell(candidate.get('license_note'))}",
        f"- Relevance: {_md_cell(candidate.get('relevance'))}",
        f"- Current decision: `{candidate.get('decision') or ''}`",
        f"- Current approval: `{candidate.get('approval_status') or 'unreviewed'}`",
        f"- Training usable: `{bool(candidate.get('training_usable'))}`",
    ]
    lines.extend([
        "",
        "## Agent Research vs Human Approval",
        "",
        "Agent/browser research may gather source facts, source URLs, screenshots/saved pages, license references, platform terms, export documentation, class-list evidence, and contradiction notes. It must not set `approved_for_training=true`, `include_in_training=true`, or any `review_items.*.approved=true` value.",
        "",
        "Agent-research tasks:",
        "",
    ])
    for task in AGENT_RESEARCH_TASKS:
        lines.append(f"- {_md_cell(task)}")
    lines.extend([
        "",
        "Human/legal approval tasks:",
        "",
    ])
    for task in HUMAN_APPROVAL_TASKS:
        lines.append(f"- {_md_cell(task)}")
    source_facts = _source_facts(candidate.get("source_facts"))
    if source_facts:
        lines.extend([
            "",
            "### Observed Source Facts",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ])
        for key in sorted(source_facts):
            value = source_facts[key]
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            lines.append(f"| `{_md_cell(key)}` | {_md_cell(value)} |")
    source_research_evidence = (
        candidate.get("source_research_evidence")
        if isinstance(candidate.get("source_research_evidence"), list)
        else []
    )
    if source_research_evidence:
        lines.extend([
            "",
            "### Current Source-Research Evidence",
            "",
            "| Finding | Evidence | Implication |",
            "| --- | --- | --- |",
        ])
        for item in source_research_evidence:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{_md_cell(item.get('finding'))} | "
                f"{_md_cell(item.get('evidence_ref'))} | "
                f"{_md_cell(item.get('implication'))} |"
            )
    suggested_mapping = (
        candidate.get("suggested_mapping")
        if isinstance(candidate.get("suggested_mapping"), dict)
        else {}
    )
    local_mapping = (
        suggested_mapping.get("local_class_to_source_labels")
        if isinstance(suggested_mapping.get("local_class_to_source_labels"), dict)
        else {}
    )
    if suggested_mapping:
        lines.extend([
            "",
            "## Suggested Class Mapping",
            "",
            "These are non-approving hints generated from the observed source labels. The reviewer must confirm or replace them in the approved import manifest.",
            "",
            "| Local Class / Bucket | Source Labels |",
            "| --- | --- |",
        ])
        for local_class in sorted(local_mapping):
            labels = local_mapping[local_class]
            if isinstance(labels, list):
                labels = ", ".join(str(label) for label in labels)
            lines.append(f"| `{_md_cell(local_class)}` | {_md_cell(labels)} |")
        hard_negative_labels = suggested_mapping.get("hard_negative_source_labels") or []
        if hard_negative_labels:
            lines.append(
                f"| `hard_negative` | {_md_cell(', '.join(str(label) for label in hard_negative_labels))} |"
            )
        ignored_labels = suggested_mapping.get("ignored_or_unmapped_source_labels") or []
        if ignored_labels:
            lines.append(
                f"| `ignored_or_unmapped` | {_md_cell(', '.join(str(label) for label in ignored_labels))} |"
            )
        mapping_warnings = suggested_mapping.get("warnings") or []
        if mapping_warnings:
            lines.extend([
                "",
                "Mapping warnings:",
                "",
            ])
            for warning in mapping_warnings:
                lines.append(f"- {_md_cell(warning)}")
    legal_references = (
        candidate.get("legal_references")
        if isinstance(candidate.get("legal_references"), dict)
        else {}
    )
    platform_references = (
        legal_references.get("platform_references")
        if isinstance(legal_references.get("platform_references"), dict)
        else {}
    )
    if legal_references:
        lines.extend([
            "",
            "## Legal References",
            "",
            "| Reference | URL |",
            "| --- | --- |",
            f"| Source page | {_md_cell(legal_references.get('source_url'))} |",
            f"| License reference | {_md_cell(legal_references.get('license_reference_url'))} |",
        ])
        for key in sorted(platform_references):
            lines.append(f"| `{_md_cell(key)}` | {_md_cell(platform_references[key])} |")
        lines.extend([
            "",
            _md_cell(legal_references.get("review_note")),
        ])
    lines.extend([
        "",
        "## Review Focus",
        "",
        _md_cell(candidate.get("review_focus")),
        "",
        "## Current Blocker",
        "",
        f"`{candidate.get('blocker') or 'none'}`",
        "",
        "## Required Review Evidence",
        "",
        "| Item | Required Evidence |",
        "| --- | --- |",
    ])
    for item in sorted(REQUIRED_REVIEW_ITEMS):
        lines.append(f"| `{item}` | {_md_cell(REVIEW_ITEM_GUIDANCE[item])} |")
    lines.extend([
        "",
        "## Evidence Bundle",
        "",
    ])
    if evidence_template:
        lines.extend([
            f"- Template path: `{evidence_template.get('path') or ''}`",
            f"- Current template SHA-256: `{evidence_template.get('sha256') or ''}`",
            "- Generated templates prefill `review_items.*.evidence_ref` with agent-collected hints, but every item remains `approved: false`.",
            "- Before approval, the reviewer must verify each hint and remove or set `prefilled_by_agent: false`; approved evidence with `prefilled_by_agent: true` is rejected.",
            "- Fill every `review_items.*` entry with `approved: true`, a concrete `evidence_ref`, and notes before using it in the checklist.",
        ])
    else:
        lines.append("- Regenerate with `--review-evidence-template-dir` to create the matching evidence bundle template.")
    lines.extend([
        "",
        "## Import Contract",
        "",
        f"- Required local classes for this capability: `{required_classes}`",
        "- `manifest_import_path` must identify the approved import manifest entry.",
        "- `raw_export_ref` must be a remote immutable export reference such as `s3://`, `gs://`, `az://`, `oci://`, `hf://`, `roboflow://`, or `https://`.",
        "- `raw_export_sha256` must be a 64-character SHA-256 digest for the reviewed export artifact.",
        "- `raw_export_local_path` must point to the reviewed local YOLO export ZIP; validation checks the ZIP SHA-256, `data.yaml` classes, mapped source labels, train/valid/test images, train/valid/test labels, matching image files for every label file, and required local-class label-file counts.",
        "- `export_format` must be `yolo` for the current training path.",
        "- Expected counts must be nonzero for every required local class before `include_in_training=true`.",
        "",
        "## Approval Path",
        "",
        "1. Fill this source's review-evidence YAML with evidence refs for every required review item.",
        "2. Recompute the evidence YAML SHA-256 and copy the path/digest into `apron_harness_seed_source_review_checklist.csv`.",
        "3. Apply the checklist to a reviewed model-pack YAML with `scripts/apron_harness_seed_source_doctor.py --apply-review-checklist-csv`.",
        "4. Fill and validate a seed import manifest before adding any exported public seed clips to the capture manifest.",
        "5. Keep production PPE blocked until closed-set promotion reports and the factory PPE Jetson full gate pass.",
        "",
    ])
    return "\n".join(lines)


def write_review_packets(report: dict[str, Any], packet_dir: Path) -> list[dict[str, Any]]:
    """Write one human-readable review packet per seed-source candidate."""
    packet_dir.mkdir(parents=True, exist_ok=True)
    evidence_templates = _review_file_lookup(report, "review_evidence_templates")
    written: list[dict[str, Any]] = []
    reviewable_candidates = [
        candidate
        for candidate in sorted(report.get("candidates") or [], key=_candidate_sort_key)
        if _is_reviewable_source_candidate(candidate)
    ]
    keep_filenames = {
        f"{_slug(candidate.get('source_ref'))}__{_slug(candidate.get('capability'))}.review_packet.md"
        for candidate in reviewable_candidates
    }
    _prune_stale_generated_files(packet_dir, "*.review_packet.md", keep_filenames)
    for candidate in reviewable_candidates:
        source_ref = str(candidate.get("source_ref") or "")
        capability = str(candidate.get("capability") or "")
        filename = f"{_slug(source_ref)}__{_slug(capability)}.review_packet.md"
        path = packet_dir / filename
        path.write_text(
            build_review_packet(
                candidate,
                evidence_templates.get((source_ref, capability)),
            ),
            encoding="utf-8",
        )
        written.append(
            {
                "source_ref": source_ref,
                "capability": capability,
                "path": _rel(path),
                "sha256": _sha256_file(path),
            }
        )
    return written


def apply_review_checklist_to_model_packs(
    *,
    model_packs_path: Path,
    checklist_csv_path: Path,
) -> dict[str, Any]:
    """Return a model-packs YAML document updated from a filled review checklist CSV."""
    errors: list[str] = []
    warnings: list[str] = []
    doc = _load_yaml(model_packs_path)
    try:
        csv_text = checklist_csv_path.read_text(encoding="utf-8")
        reader = csv.DictReader(csv_text.splitlines())
        rows = list(reader)
    except Exception as exc:
        return {
            "ok": False,
            "path": _rel(checklist_csv_path),
            "updated_model_packs": None,
            "errors": [f"review checklist CSV unreadable: {exc}"],
            "warnings": warnings,
        }

    fieldnames = reader.fieldnames or []
    missing_fields = [field for field in REVIEW_CHECKLIST_FIELDS if field not in fieldnames]
    if missing_fields:
        errors.append("review checklist CSV missing required columns: " + ", ".join(missing_fields))

    factory_pack = ((doc.get("packs") or {}).get("factory_ppe_3cam") or {})
    shared_sources = doc.get("shared_sources") if isinstance(doc.get("shared_sources"), dict) else {}
    sourcing = factory_pack.get("sourcing_status") if isinstance(factory_pack.get("sourcing_status"), dict) else {}
    candidates = sourcing.get("candidate_sources") if isinstance(sourcing.get("candidate_sources"), list) else []
    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = (str(candidate.get("source_ref") or ""), str(candidate.get("capability") or ""))
        candidates_by_key[key] = candidate

    seen_keys: set[tuple[str, str]] = set()
    applied_count = 0
    approved_count = 0
    for row_number, row in enumerate(rows, start=2):
        source_ref = str(row.get("source_ref") or "").strip()
        capability = str(row.get("capability") or "").strip()
        key = (source_ref, capability)
        prefix = f"line {row_number}"
        row_errors: list[str] = []
        if not source_ref:
            row_errors.append("source_ref is required")
        if capability not in ALLOWED_CAPABILITIES:
            row_errors.append(f"capability must be one of {sorted(ALLOWED_CAPABILITIES)}")
        if key in seen_keys:
            row_errors.append(f"duplicate source_ref/capability row: {source_ref}/{capability}")
        seen_keys.add(key)
        candidate = candidates_by_key.get(key)
        source = shared_sources.get(source_ref) if isinstance(shared_sources.get(source_ref), dict) else {}
        if candidate is None:
            row_errors.append(f"source_ref/capability pair is not in model packs: {source_ref}/{capability}")
        else:
            if str(row.get("source_url") or "").strip() != str(source.get("url") or "").strip():
                row_errors.append("source_url must match model-packs shared source")
            if str(row.get("license_note") or "").strip() != str(source.get("license_note") or "").strip():
                row_errors.append("license_note must match model-packs shared source")

        approval_status = str(row.get("approval_status") or "unreviewed").strip()
        if approval_status not in ALLOWED_APPROVAL_STATUS:
            row_errors.append(f"invalid approval_status: {approval_status or 'missing'}")
        approved_for_training = _parse_bool(row.get("approved_for_training"))
        if approved_for_training is None:
            row_errors.append("approved_for_training must be boolean")
            approved_for_training = False
        training_usable = _parse_bool(row.get("training_usable"))
        if training_usable is None:
            row_errors.append("training_usable must be boolean")
            training_usable = False

        completed_review: dict[str, bool] = {}
        for field in sorted(REVIEW_BOOLEAN_FIELDS):
            parsed = _parse_bool(row.get(field))
            if parsed is None:
                row_errors.append(f"{field} must be boolean")
                parsed = False
            completed_review[field] = parsed

        blocker = str(row.get("current_blocker") or "").strip()
        reviewed_by = str(row.get("reviewed_by") or "").strip()
        reviewed_at = str(row.get("reviewed_at") or "").strip()
        manifest_import_path = str(row.get("manifest_import_path") or "").strip()
        review_evidence_path = str(row.get("review_evidence_path") or "").strip()
        review_evidence_sha256 = str(row.get("review_evidence_sha256") or "").strip()
        if approved_for_training:
            if approval_status != "approved_for_training":
                row_errors.append("approved_for_training=true requires approval_status=approved_for_training")
            if training_usable is not True:
                row_errors.append("approved_for_training=true requires training_usable=true")
            missing_reviews = sorted(
                field for field, approved in completed_review.items() if approved is not True
            )
            if missing_reviews:
                row_errors.append("completed review approvals missing: " + ", ".join(missing_reviews))
            if blocker:
                row_errors.append("approved source must clear current_blocker")
            if not reviewed_by:
                row_errors.append("reviewed_by is required for approved sources")
            if not reviewed_at:
                row_errors.append("reviewed_at is required for approved sources")
            if not manifest_import_path:
                row_errors.append("manifest_import_path is required for approved sources")
            row_errors.extend(
                error + " for approved sources"
                for error in _review_evidence_errors(
                    review_evidence_path,
                    review_evidence_sha256,
                    source_ref=source_ref,
                    capability=capability,
                    source_url=str(source.get("url") or ""),
                    license_note=str(source.get("license_note") or ""),
                    reviewed_by=reviewed_by,
                    reviewed_at=reviewed_at,
                )
            )
        if not approved_for_training and review_evidence_sha256:
            review_evidence_sha_error = _sha256_error(
                review_evidence_sha256,
                "review_evidence_sha256",
            )
            if review_evidence_sha_error:
                row_errors.append(review_evidence_sha_error)
        if not approved_for_training and approval_status == "approved_for_training":
            row_errors.append("approval_status=approved_for_training requires approved_for_training=true")

        if row_errors:
            errors.extend(f"{prefix}: {error}" for error in row_errors)
            continue
        if candidate is None:
            continue

        candidate["approval_status"] = approval_status
        candidate["approved_for_training"] = approved_for_training
        candidate["completed_review"] = completed_review
        if reviewed_by:
            candidate["reviewed_by"] = reviewed_by
        else:
            candidate.pop("reviewed_by", None)
        if reviewed_at:
            candidate["reviewed_at"] = reviewed_at
        else:
            candidate.pop("reviewed_at", None)
        if manifest_import_path:
            candidate["manifest_import_path"] = manifest_import_path
        else:
            candidate.pop("manifest_import_path", None)
        if review_evidence_path:
            candidate["review_evidence_path"] = review_evidence_path
        else:
            candidate.pop("review_evidence_path", None)
        if review_evidence_sha256:
            candidate["review_evidence_sha256"] = review_evidence_sha256
        else:
            candidate.pop("review_evidence_sha256", None)
        if row.get("review_notes"):
            candidate["review_notes"] = str(row.get("review_notes") or "").strip()
        if approved_for_training:
            candidate["blocker"] = ""
            approved_count += 1
        elif blocker:
            candidate["blocker"] = blocker
        applied_count += 1

    updated_audit = _audit_seed_sources_doc(doc, model_packs_path)
    if updated_audit.get("ok") is not True:
        errors.extend(str(error) for error in updated_audit.get("errors") or [])
    return {
        "ok": not errors,
        "path": _rel(checklist_csv_path),
        "row_count": len(rows),
        "applied_count": applied_count,
        "approved_count": approved_count,
        "updated_audit": updated_audit,
        "updated_model_packs": doc if not errors else None,
        "errors": errors,
        "warnings": warnings,
    }


def validate_import_manifest(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {
            "ok": False,
            "gate_passed": False,
            "path": _rel(path),
            "errors": [f"seed import manifest unreadable: {exc}"],
            "blockers": [],
            "imports": [],
        }

    if doc.get("version") != 1:
        errors.append("version must be 1")
    if doc.get("kind") != "apron_harness_seed_import_manifest":
        errors.append("kind must be apron_harness_seed_import_manifest")
    raw_imports = doc.get("imports")
    if not isinstance(raw_imports, list) or not raw_imports:
        errors.append("imports must contain at least one source import entry")
        raw_imports = []

    candidates_by_key = {
        (str(item.get("source_ref") or ""), str(item.get("capability") or "")): item
        for item in report.get("candidates") or []
        if isinstance(item, dict)
    }
    import_reports: list[dict[str, Any]] = []
    included_count = 0
    approved_count = 0
    for index, raw in enumerate(raw_imports):
        prefix = f"imports[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} must be an object")
            continue
        source_ref = str(raw.get("source_ref") or "")
        capability = str(raw.get("capability") or "")
        include_in_training = bool(raw.get("include_in_training"))
        candidate = candidates_by_key.get((source_ref, capability))
        item_errors: list[str] = []
        item_blockers: list[str] = []
        yolo_export_preflight: dict[str, Any] = {"checked": False}
        review_artifact_preflight: dict[str, Any] = {"checked": False, "errors": []}
        if not source_ref:
            item_errors.append("source_ref is required")
        if capability not in ALLOWED_CAPABILITIES:
            item_errors.append(f"capability must be one of {sorted(ALLOWED_CAPABILITIES)}")
        if candidate is None:
            item_errors.append("source_ref/capability pair is not in seed source review")

        completed_review = raw.get("completed_review") if isinstance(raw.get("completed_review"), dict) else {}
        missing_reviews = sorted(
            item for item in REVIEW_BOOLEAN_FIELDS if completed_review.get(item) is not True
        )
        if include_in_training:
            included_count += 1
            review_artifact_preflight = _review_artifact_preflight(
                raw.get("review_artifacts"),
                source_ref=source_ref,
                capability=capability,
                report=report,
            )
            item_errors.extend(review_artifact_preflight["errors"])
            if candidate and candidate.get("training_usable") is not True:
                item_blockers.append(f"{source_ref}: seed source review is not training-usable")
            if str(raw.get("review_status") or "") != "approved_for_training":
                item_errors.append("review_status must be approved_for_training when include_in_training=true")
            if not raw.get("reviewed_by"):
                item_errors.append("reviewed_by is required when include_in_training=true")
            if not raw.get("reviewed_at"):
                item_errors.append("reviewed_at is required when include_in_training=true")
            if candidate and raw.get("reviewed_by") != candidate.get("reviewed_by"):
                item_errors.append("reviewed_by must match the approved seed source review")
            if candidate and raw.get("reviewed_at") != candidate.get("reviewed_at"):
                item_errors.append("reviewed_at must match the approved seed source review")
            if not raw.get("manifest_import_path"):
                item_errors.append("manifest_import_path is required when include_in_training=true")
            if candidate and raw.get("manifest_import_path") != candidate.get("manifest_import_path"):
                item_errors.append("manifest_import_path must match the approved seed source review")
            raw_export_ref_error = _raw_export_ref_error(raw.get("raw_export_ref"))
            if raw_export_ref_error:
                item_errors.append(raw_export_ref_error)
            raw_export_sha256_error = _raw_export_sha256_error(raw.get("raw_export_sha256"))
            if raw_export_sha256_error:
                item_errors.append(raw_export_sha256_error)
            if str(raw.get("export_format") or "") not in ALLOWED_IMPORT_EXPORT_FORMATS:
                item_errors.append(
                    "export_format must be one of "
                    + ", ".join(sorted(ALLOWED_IMPORT_EXPORT_FORMATS))
                    + " when include_in_training=true"
                )
            if missing_reviews:
                item_errors.append("completed_review missing approvals: " + ", ".join(missing_reviews))
            class_mapping = raw.get("class_mapping") if isinstance(raw.get("class_mapping"), dict) else {}
            if not class_mapping:
                item_errors.append("class_mapping is required when include_in_training=true")
            else:
                mapped_classes = _normalized_mapping_values(class_mapping)
                missing_mapped_classes = sorted(
                    REQUIRED_IMPORT_COUNT_CLASSES.get(capability, set()) - mapped_classes
                )
                if missing_mapped_classes:
                    item_errors.append(
                        "class_mapping must map source classes to required local classes: "
                        + ", ".join(missing_mapped_classes)
                    )
            if not raw.get("person_box_policy"):
                item_errors.append("person_box_policy is required when include_in_training=true")
            if not raw.get("hard_negative_policy"):
                item_errors.append("hard_negative_policy is required when include_in_training=true")
            split_plan = raw.get("split_plan") if isinstance(raw.get("split_plan"), dict) else {}
            for split in ["train", "val", "test"]:
                if not split_plan.get(split):
                    item_errors.append(f"split_plan.{split} is required when include_in_training=true")
            expected_counts = (
                raw.get("expected_labeled_images_per_class")
                if isinstance(raw.get("expected_labeled_images_per_class"), dict)
                else {}
            )
            if not expected_counts:
                item_errors.append(
                    "expected_labeled_images_per_class is required when include_in_training=true"
                )
            for class_name in sorted(REQUIRED_IMPORT_COUNT_CLASSES.get(capability, set())):
                if _count_value(expected_counts, class_name) <= 0:
                    item_errors.append(
                        f"expected_labeled_images_per_class.{class_name} must be greater than 0 "
                        "when include_in_training=true"
                    )
            export_format = str(raw.get("export_format") or "")
            if export_format in {"", "yolo"}:
                yolo_export_preflight, export_errors = _validate_yolo_export_archive(
                    raw.get("raw_export_local_path"),
                    expected_sha256=raw.get("raw_export_sha256"),
                    class_mapping=class_mapping,
                    capability=capability,
                    expected_counts=expected_counts,
                )
                item_errors.extend(export_errors)
            if not item_errors and not item_blockers:
                approved_count += 1
        else:
            item_blockers.append(f"{source_ref}: include_in_training=false")

        errors.extend(f"{prefix}.{error}" for error in item_errors)
        blockers.extend(f"{prefix}.{blocker}" for blocker in item_blockers)
        import_reports.append(
            {
                "source_ref": source_ref,
                "capability": capability,
                "include_in_training": include_in_training,
                "approved_for_training": include_in_training and not item_errors and not item_blockers,
                "manifest_import_path": raw.get("manifest_import_path"),
                "raw_export_ref": raw.get("raw_export_ref"),
                "raw_export_sha256": raw.get("raw_export_sha256"),
                "raw_export_local_path": raw.get("raw_export_local_path"),
                "export_format": raw.get("export_format"),
                "yolo_export_preflight": yolo_export_preflight,
                "review_artifact_preflight": review_artifact_preflight,
                "reviewed_by": raw.get("reviewed_by"),
                "reviewed_at": raw.get("reviewed_at"),
                "errors": item_errors,
                "blockers": item_blockers,
            }
        )

    if included_count > 0:
        source_review_sha256 = str(doc.get("source_review_sha256") or "").strip()
        if not source_review_sha256:
            errors.append("source_review_sha256 is required when include_in_training=true")
        elif source_review_sha256 != source_review_fingerprint(report):
            errors.append("source_review_sha256 does not match seed source review")

    gate_passed = not errors and not blockers and included_count > 0 and approved_count == included_count
    if not errors and included_count == 0:
        blockers.append("no seed imports are approved for training")
    return {
        "ok": not errors,
        "gate_passed": gate_passed,
        "path": _rel(path),
        "source_review_sha256": doc.get("source_review_sha256"),
        "source_review_sha256_matches": doc.get("source_review_sha256") == source_review_fingerprint(report),
        "included_count": included_count,
        "approved_count": approved_count,
        "imports": import_reports,
        "errors": errors,
        "blockers": blockers,
    }


def render_work_order(report: dict[str, Any]) -> str:
    lines = [
        "# Apron/Harness Public Seed Source Review",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        f"- Gate: `{'pass' if report.get('gate_passed') else 'blocked'}`",
        f"- Status: `{report.get('status')}`",
        f"- Candidate sources: `{report.get('candidate_count', 0)}`",
        f"- Training-usable sources: `{report.get('training_usable_count', 0)}`",
        "",
        "## Priority Review Queue",
        "",
        "| Capability | Top Sources | Training-usable |",
        "| --- | --- | --- |",
    ]
    queue_summary = report.get("review_queue_summary") if isinstance(report.get("review_queue_summary"), dict) else {}
    capabilities = queue_summary.get("capabilities") if isinstance(queue_summary.get("capabilities"), dict) else {}
    for capability in sorted(ALLOWED_CAPABILITIES):
        capability_summary = capabilities.get(capability) if isinstance(capabilities.get(capability), dict) else {}
        top_sources = capability_summary.get("top_review_candidates") or []
        source_list = ", ".join(
            f"`{item.get('source_ref')}` ({item.get('review_priority')})"
            for item in top_sources
            if isinstance(item, dict)
        )
        lines.append(
            "| "
            f"`{capability}` | "
            f"{source_list or 'none'} | "
            f"`{capability_summary.get('training_usable_count', 0)}` / "
            f"`{capability_summary.get('candidate_count', 0)}` |"
        )
    research = (
        report.get("source_research_readiness")
        if isinstance(report.get("source_research_readiness"), dict)
        else {}
    )
    if research:
        lines.extend([
            "",
            "## Source Research Readiness",
            "",
            "This section only tracks agent/browser evidence completeness. It does not approve any "
            "source for training or import.",
            "",
            f"- Evidence-ready for human review: `{research.get('evidence_ready_count', 0)}` / "
            f"`{research.get('candidate_count', 0)}`",
            f"- Still needing agent research: `{research.get('needs_agent_research_count', 0)}`",
            f"- Training-usable sources: `{research.get('training_usable_count', 0)}`",
            "",
            "### Evidence-Ready Sources",
            "",
            "| Priority | Source | Capability | Evidence Items | Next Action |",
            "| --- | --- | --- | --- | --- |",
        ])
        ready_sources = research.get("evidence_ready_sources") if isinstance(research.get("evidence_ready_sources"), list) else []
        if ready_sources:
            for item in ready_sources:
                if not isinstance(item, dict):
                    continue
                status = (
                    item.get("source_research_status")
                    if isinstance(item.get("source_research_status"), dict)
                    else {}
                )
                lines.append(
                    "| "
                    f"`{item.get('review_priority')}` | "
                    f"`{_md_cell(item.get('source_ref'))}` | "
                    f"`{_md_cell(item.get('capability'))}` | "
                    f"`{status.get('evidence_count', 0)}` | "
                    f"{_md_cell(status.get('next_action'))} |"
                )
        else:
            lines.append("| n/a | none | n/a | `0` | collect_source_page_export_terms_and_license_evidence |")
        lines.extend([
            "",
            "### Research-Closed Sources",
            "",
            "| Priority | Source | Capability | Status | Next Action |",
            "| --- | --- | --- | --- | --- |",
        ])
        closed_sources = (
            research.get("source_research_closed_sources")
            if isinstance(research.get("source_research_closed_sources"), list)
            else []
        )
        if closed_sources:
            for item in closed_sources:
                if not isinstance(item, dict):
                    continue
                status = (
                    item.get("source_research_status")
                    if isinstance(item.get("source_research_status"), dict)
                    else {}
                )
                lines.append(
                    "| "
                    f"`{item.get('review_priority')}` | "
                    f"`{_md_cell(item.get('source_ref'))}` | "
                    f"`{_md_cell(item.get('capability'))}` | "
                    f"`{_md_cell(status.get('status'))}` | "
                    f"{_md_cell(status.get('next_action'))} |"
                )
        else:
            lines.append("| n/a | none | n/a | none | n/a |")
        lines.extend([
            "",
            "### Next Agent Research Sources",
            "",
            "| Priority | Source | Capability | Missing | Next Action |",
            "| --- | --- | --- | --- | --- |",
        ])
        research_sources = (
            research.get("next_agent_research_sources")
            if isinstance(research.get("next_agent_research_sources"), list)
            else []
        )
        if research_sources:
            for item in research_sources:
                if not isinstance(item, dict):
                    continue
                status = (
                    item.get("source_research_status")
                    if isinstance(item.get("source_research_status"), dict)
                    else {}
                )
                missing = ", ".join(str(value) for value in status.get("missing") or [])
                lines.append(
                    "| "
                    f"`{item.get('review_priority')}` | "
                    f"`{_md_cell(item.get('source_ref'))}` | "
                    f"`{_md_cell(item.get('capability'))}` | "
                    f"{_md_cell(missing)} | "
                    f"{_md_cell(status.get('next_action'))} |"
                )
        else:
            lines.append("| n/a | none | n/a | none | human_legal_review_and_seed_import_manifest |")
    lines.extend([
        "",
        "## Source Candidates",
        "",
        "| Priority | Source | Capability | Approval | Training | Review Focus | URL | Blocker |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for item in sorted(report.get("candidates") or [], key=_candidate_sort_key):
        lines.append(
            "| "
            f"`{item.get('review_priority')}` | "
            f"`{item.get('source_ref')}` | "
            f"`{item.get('capability')}` | "
            f"`{item.get('approval_status')}` | "
            f"`{bool(item.get('training_usable'))}` | "
            f"{_md_cell(item.get('review_focus'))} | "
            f"{_md_cell(item.get('url'))} | "
            f"{_md_cell(item.get('blocker'))} |"
        )
    lines.extend([
        "",
        "## Required Review",
        "",
        "Each source must remain out of training until all of these are recorded:",
        "",
    ])
    for item in sorted(REQUIRED_REVIEW_ITEMS):
        lines.append(f"- `{item}`")
    lines.extend([
        "",
        "A source can become training-usable only when `approval_status=approved_for_training`, "
        "`approved_for_training=true`, `manifest_import_path`, `reviewed_by`, `reviewed_at`, "
        "`review_evidence_path`, `review_evidence_sha256`, and `completed_review` approvals for every "
        "required review item are present. The review evidence path must point to a local file and the "
        "SHA-256 must match that file. The file must be a YAML/JSON "
        "`apron_harness_seed_source_review_evidence` document with matching source/capability/reviewer "
        "metadata and approved evidence refs for every required review item.",
        "",
    ])
    coverage_plan = report.get("source_coverage_plan") if isinstance(report.get("source_coverage_plan"), dict) else {}
    if coverage_plan:
        lines.extend([
            "## Source Coverage Plan",
            "",
            "The source coverage plan is generated from non-approving suggested mappings. It is a "
            "review-prioritization artifact only; it does not verify labels, approve licenses, or "
            "authorize training import.",
            "",
            f"- Path: `{coverage_plan.get('path') or ''}`",
            f"- SHA-256: `{coverage_plan.get('sha256') or ''}`",
            f"- Candidate count: `{coverage_plan.get('candidate_count', 0)}`",
            f"- Coverage gaps: `{coverage_plan.get('coverage_gap_count', 0)}`",
            "",
        ])
        reconciliation = (
            coverage_plan.get("person_box_reconciliation")
            if isinstance(coverage_plan.get("person_box_reconciliation"), dict)
            else {}
        )
        if reconciliation:
            lines.extend([
                "### Person-Box Reconciliation",
                "",
                "| Capability | Status | Candidate Person Sources | Candidate Sources Missing Person |",
                "| --- | --- | ---: | ---: |",
            ])
            for capability, item in sorted(reconciliation.items()):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "| "
                    f"`{_md_cell(capability)}` | "
                    f"`{_md_cell(item.get('status'))}` | "
                    f"`{item.get('candidate_person_source_count', 0)}` | "
                    f"`{item.get('candidate_without_person_source_count', 0)}` |"
                )
            lines.append("")
    template_rows = _review_template_rows(report)
    if template_rows:
        lines.extend([
            "## Review Evidence Templates",
            "",
            "The generated template files below are non-approving starting points. Fill the matching "
            "template for each reviewed source, then put the filled file path and fresh SHA-256 in "
            "`apron_harness_seed_source_review_checklist.csv`.",
            "",
            "| Priority | Source | Capability | Template | SHA-256 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in template_rows:
            lines.append(
                "| "
                f"`{row['review_priority']}` | "
                f"`{_md_cell(row['source_ref'])}` | "
                f"`{_md_cell(row['capability'])}` | "
                f"`{_md_cell(row['path'])}` | "
                f"`{_md_cell(row['sha256'])}` |"
            )
        lines.extend([
            "",
            "After editing a template, recompute its digest, for example:",
            "",
            "```bash",
            "shasum -a 256 /path/to/filled.review_evidence.yaml",
            "```",
            "",
        ])
    else:
        lines.extend([
            "## Review Evidence Templates",
            "",
            "Regenerate this work order with `--review-evidence-template-dir "
            "qa/video_eval/results/apron_harness_seed_source_review_evidence` to write fillable "
            "review-evidence YAML files and list them here.",
            "",
        ])
    packet_rows = _review_packet_rows(report)
    if packet_rows:
        lines.extend([
            "## Review Packets",
            "",
            "Use these generated packets as the reviewer-facing source dossiers. They summarize the "
            "source facts, current blockers, required evidence, and import contract for each candidate.",
            "",
            "| Priority | Source | Capability | Packet | SHA-256 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in packet_rows:
            lines.append(
                "| "
                f"`{row['review_priority']}` | "
                f"`{_md_cell(row['source_ref'])}` | "
                f"`{_md_cell(row['capability'])}` | "
                f"`{_md_cell(row['path'])}` | "
                f"`{_md_cell(row['sha256'])}` |"
            )
        lines.append("")
    else:
        lines.extend([
            "## Review Packets",
            "",
            "Regenerate this work order with `--review-packet-dir "
            "qa/video_eval/results/apron_harness_seed_source_review_packets` to write reviewer-facing "
            "source packets and list them here.",
            "",
        ])
    next_batch = report.get("next_review_batch") if isinstance(report.get("next_review_batch"), dict) else {}
    if next_batch:
        lines.extend([
            "## Next Review Batch",
            "",
            "Use this JSON file as the browser/agent handoff for source research. It lists the next "
            "priority sources and the packet, evidence-template, checklist, and import-template paths. "
            "It is explicitly non-approving.",
            "",
            f"- Path: `{next_batch.get('path') or ''}`",
            f"- SHA-256: `{next_batch.get('sha256') or ''}`",
            f"- Selected sources: `{next_batch.get('selected_count', 0)}`",
            "",
        ])
    else:
        lines.extend([
            "## Next Review Batch",
            "",
            "Regenerate this work order with `--next-review-batch-out "
            "qa/video_eval/results/apron_harness_next_source_review_batch.json` to write the "
            "non-approving browser/agent source-review handoff.",
            "",
        ])
    kickoff = report.get("review_kickoff") if isinstance(report.get("review_kickoff"), dict) else {}
    if kickoff:
        lines.extend([
            "## Source Review Kickoff",
            "",
            "Use this concise Markdown file as the first operator-facing handoff before opening the "
            "full packets and checklist. It is explicitly non-approving.",
            "",
            f"- Path: `{kickoff.get('path') or ''}`",
            f"- SHA-256: `{kickoff.get('sha256') or ''}`",
            f"- Selected sources: `{kickoff.get('selected_count', 0)}`",
            "",
        ])
    else:
        lines.extend([
            "## Source Review Kickoff",
            "",
            "Regenerate this work order with `--review-kickoff-out "
            "qa/video_eval/results/apron_harness_source_review_kickoff.md` to write the concise "
            "non-approving operator kickoff.",
            "",
        ])
    lines.extend([
        "## Follow-Up Commands",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        "  --review-bundle-out \"\" \\",
        "  --validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json",
        "",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        "  --review-bundle-out \"\" \\",
        "  --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
        "",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        "  --model-packs qa/video_eval/model_packs.yaml \\",
        "  --apply-review-checklist-csv /path/to/filled/apron_harness_seed_source_review_checklist.csv \\",
        "  --updated-model-packs-out /path/to/reviewed/model_packs.yaml",
        "",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        "  --model-packs /path/to/reviewed/model_packs.yaml \\",
        "  --validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml",
        "```",
        "",
        "This writes a reviewed model-pack YAML to a new path. It does not mutate the source file in place, "
        "and approved rows are rejected unless every review checkbox is true, `training_usable=true`, "
        "the blocker is cleared, reviewer/timestamp/import-path fields are present, and a local review "
        "evidence bundle path plus matching SHA-256 is recorded.",
        "",
        "The filled import-manifest command must exit `0` and print `IMPORT_MANIFEST: gate=pass` before "
        "any public seed export is materialized. A template or partially filled manifest must remain "
        "blocked and exit nonzero with `IMPORT_MANIFEST: gate=blocked`.",
        "",
        "## Current Blockers",
        "",
    ])
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    errors = report.get("errors") or []
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in errors)
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in warnings)
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit public apron/harness seed sources before dataset import.")
    parser.add_argument("--model-packs", default=str(DEFAULT_MODEL_PACKS))
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Write JSON report")
    parser.add_argument("--work-order-out", default=str(DEFAULT_WORK_ORDER), help="Write Markdown review work order")
    parser.add_argument(
        "--import-template-out",
        default=str(DEFAULT_IMPORT_TEMPLATE),
        help="Write a fillable public seed import manifest template",
    )
    parser.add_argument(
        "--minimum-import-template-out",
        default=str(DEFAULT_MINIMUM_IMPORT_TEMPLATE),
        help="Write a fillable import manifest template limited to priority coverage sources",
    )
    parser.add_argument(
        "--review-checklist-csv-out",
        default=str(DEFAULT_REVIEW_CHECKLIST_CSV),
        help="Write a fillable public seed-source review checklist CSV",
    )
    parser.add_argument(
        "--review-evidence-template-dir",
        default=str(DEFAULT_REVIEW_EVIDENCE_TEMPLATE_DIR),
        help="Write fillable review evidence bundle templates, one YAML file per candidate source",
    )
    parser.add_argument(
        "--review-packet-dir",
        default=str(DEFAULT_REVIEW_PACKET_DIR),
        help="Write reviewer-facing Markdown packets, one file per candidate source",
    )
    parser.add_argument(
        "--next-review-batch-out",
        default=str(DEFAULT_NEXT_REVIEW_BATCH),
        help="Write a non-approving JSON handoff for the next source-review batch",
    )
    parser.add_argument(
        "--review-kickoff-out",
        default=str(DEFAULT_REVIEW_KICKOFF),
        help="Write a concise non-approving Markdown kickoff for the next source-review batch",
    )
    parser.add_argument(
        "--source-coverage-plan-out",
        default=str(DEFAULT_SOURCE_COVERAGE_PLAN),
        help="Write a non-approving JSON source coverage plan for apron/harness class review",
    )
    parser.add_argument(
        "--review-bundle-out",
        default=str(DEFAULT_REVIEW_BUNDLE),
        help="Write a non-approving JSON manifest of source-review handoff artifacts and hashes",
    )
    parser.add_argument(
        "--next-review-batch-limit",
        type=int,
        default=DEFAULT_NEXT_REVIEW_BATCH_LIMIT,
        help="Number of priority sources to include in --next-review-batch-out; use 0 for all",
    )
    parser.add_argument(
        "--validate-import-manifest",
        default="",
        help="Validate a filled public seed import manifest against the seed-source review",
    )
    parser.add_argument(
        "--validate-review-bundle",
        default="",
        help="Validate a generated source-review handoff bundle and all recorded artifact hashes",
    )
    parser.add_argument(
        "--validate-next-review-batch",
        default="",
        help="Validate a generated next source-review batch and all recorded artifact hashes",
    )
    parser.add_argument(
        "--apply-review-checklist-csv",
        default="",
        help="Apply a filled public seed-source review checklist CSV to model packs in memory",
    )
    parser.add_argument(
        "--updated-model-packs-out",
        default="",
        help="Write model packs YAML updated from --apply-review-checklist-csv",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    return parser


def _suppress_default_outputs_for_validation(args: argparse.Namespace, argv: list[str]) -> None:
    validation_requested = bool(
        args.validate_import_manifest
        or args.validate_review_bundle
        or args.validate_next_review_batch
    )
    if not validation_requested:
        return

    output_defaults = {
        "out": ("--out", DEFAULT_OUT),
        "work_order_out": ("--work-order-out", DEFAULT_WORK_ORDER),
        "import_template_out": ("--import-template-out", DEFAULT_IMPORT_TEMPLATE),
        "minimum_import_template_out": (
            "--minimum-import-template-out",
            DEFAULT_MINIMUM_IMPORT_TEMPLATE,
        ),
        "review_checklist_csv_out": ("--review-checklist-csv-out", DEFAULT_REVIEW_CHECKLIST_CSV),
        "review_evidence_template_dir": (
            "--review-evidence-template-dir",
            DEFAULT_REVIEW_EVIDENCE_TEMPLATE_DIR,
        ),
        "review_packet_dir": ("--review-packet-dir", DEFAULT_REVIEW_PACKET_DIR),
        "next_review_batch_out": ("--next-review-batch-out", DEFAULT_NEXT_REVIEW_BATCH),
        "review_kickoff_out": ("--review-kickoff-out", DEFAULT_REVIEW_KICKOFF),
        "source_coverage_plan_out": ("--source-coverage-plan-out", DEFAULT_SOURCE_COVERAGE_PLAN),
        "review_bundle_out": ("--review-bundle-out", DEFAULT_REVIEW_BUNDLE),
    }
    explicit_flags = set(argv)
    for attr, (flag, default) in output_defaults.items():
        if flag in explicit_flags:
            continue
        if getattr(args, attr) == str(default):
            setattr(args, attr, "")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    _suppress_default_outputs_for_validation(args, raw_argv)
    report = audit_seed_sources(Path(args.model_packs))
    if args.review_evidence_template_dir:
        template_dir = Path(args.review_evidence_template_dir)
        report["review_evidence_templates"] = {
            "dir": str(template_dir),
            "files": write_review_evidence_templates(report, template_dir),
        }
    if args.review_packet_dir:
        packet_dir = Path(args.review_packet_dir)
        report["review_packets"] = {
            "dir": str(packet_dir),
            "files": write_review_packets(report, packet_dir),
        }
    if args.import_template_out:
        import_template_path = Path(args.import_template_out)
        import_template_path.parent.mkdir(parents=True, exist_ok=True)
        import_template_path.write_text(
            yaml.safe_dump(
                build_import_manifest_template(report, seed_review_report=args.out),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        report["seed_import_manifest_template"] = {
            "path": _rel(import_template_path),
            "sha256": _sha256_file(import_template_path),
        }
    if args.minimum_import_template_out:
        minimum_import_template_path = Path(args.minimum_import_template_out)
        minimum_import_template_path.parent.mkdir(parents=True, exist_ok=True)
        minimum_source_refs = minimum_approval_source_refs(report)
        minimum_import_template_path.write_text(
            yaml.safe_dump(
                build_import_manifest_template(
                    report,
                    seed_review_report=args.out,
                    source_refs=minimum_source_refs,
                    template_scope="minimum_priority_coverage_sources",
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        report["minimum_seed_import_manifest_template"] = {
            "path": _rel(minimum_import_template_path),
            "sha256": _sha256_file(minimum_import_template_path),
            "source_refs": sorted(minimum_source_refs),
        }
    if args.review_checklist_csv_out:
        checklist_path = Path(args.review_checklist_csv_out)
        checklist_path.parent.mkdir(parents=True, exist_ok=True)
        checklist_path.write_text(build_review_checklist_csv(report), encoding="utf-8")
        report["review_checklist_csv"] = {
            "path": _rel(checklist_path),
            "sha256": _sha256_file(checklist_path),
        }
    if args.apply_review_checklist_csv:
        checklist_apply = apply_review_checklist_to_model_packs(
            model_packs_path=Path(args.model_packs),
            checklist_csv_path=Path(args.apply_review_checklist_csv),
        )
        report["review_checklist_apply"] = {
            key: value
            for key, value in checklist_apply.items()
            if key != "updated_model_packs"
        }
        if not checklist_apply["ok"]:
            report["errors"].extend(checklist_apply["errors"])
            report["ok"] = False
        elif args.updated_model_packs_out:
            updated_path = Path(args.updated_model_packs_out)
            updated_path.parent.mkdir(parents=True, exist_ok=True)
            updated_path.write_text(
                yaml.safe_dump(checklist_apply["updated_model_packs"], sort_keys=False),
                encoding="utf-8",
            )
            report["review_checklist_apply"]["updated_model_packs_out"] = str(updated_path)
        elif args.apply_review_checklist_csv:
            report["warnings"].append(
                "--apply-review-checklist-csv validated the update but no --updated-model-packs-out was provided"
            )
    source_review_validation_report = copy.deepcopy(report)
    if args.validate_import_manifest:
        import_review = validate_import_manifest(Path(args.validate_import_manifest), report)
        report["import_manifest_review"] = import_review
        if not import_review["ok"]:
            report["errors"].extend(import_review["errors"])
            report["ok"] = False
        if not import_review["gate_passed"]:
            report["gate_passed"] = False
            report["status"] = "blocked_pending_import_manifest"
            report["blockers"].extend(import_review["blockers"])
            report["ok"] = False
    report["review_queue"] = build_review_queue(report)
    if args.validate_next_review_batch:
        batch_validation = validate_next_review_batch(
            Path(args.validate_next_review_batch),
            source_review_validation_report,
        )
        report["next_review_batch_validation"] = batch_validation
        if not batch_validation["ok"]:
            report["errors"].extend(batch_validation["errors"])
            report["ok"] = False
    if args.next_review_batch_out:
        report["next_review_batch"] = write_next_review_batch(
            report,
            Path(args.next_review_batch_out),
            limit=args.next_review_batch_limit,
            source_review_report=args.out,
        )
    if args.review_kickoff_out:
        report["review_kickoff"] = write_review_kickoff(
            report,
            Path(args.review_kickoff_out),
            limit=args.next_review_batch_limit,
            source_review_report=args.out,
        )
    if args.source_coverage_plan_out:
        report["source_coverage_plan"] = write_source_coverage_plan(
            report,
            Path(args.source_coverage_plan_out),
            source_review_report=args.out,
        )
    if args.work_order_out:
        work_order_path = Path(args.work_order_out)
        work_order_path.parent.mkdir(parents=True, exist_ok=True)
        work_order_path.write_text(render_work_order(report), encoding="utf-8")
        report["work_order"] = {
            "path": _rel(work_order_path),
            "sha256": _sha256_file(work_order_path),
        }
    if args.validate_review_bundle:
        bundle_validation = validate_review_bundle_manifest(
            Path(args.validate_review_bundle),
            source_review_validation_report,
        )
        report["review_bundle_validation"] = bundle_validation
        if not bundle_validation["ok"]:
            report["errors"].extend(bundle_validation["errors"])
            report["ok"] = False
    if args.review_bundle_out:
        report["review_bundle"] = write_review_bundle_manifest(
            report,
            Path(args.review_bundle_out),
            source_review_report=args.out,
        )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        print(
            f"{status}: gate={'pass' if report['gate_passed'] else 'blocked'} "
            f"candidates={report['candidate_count']} "
            f"training_usable={report['training_usable_count']}"
        )
        if args.out:
            print(f"wrote: {args.out}")
        if args.work_order_out:
            print(f"wrote: {args.work_order_out}")
        if args.import_template_out:
            print(f"wrote: {args.import_template_out}")
        if args.minimum_import_template_out:
            minimum_template = report.get("minimum_seed_import_manifest_template") or {}
            print(
                "wrote: "
                f"{args.minimum_import_template_out} "
                f"({len(minimum_template.get('source_refs') or [])} minimum sources)"
            )
        if args.review_checklist_csv_out:
            print(f"wrote: {args.review_checklist_csv_out}")
        if args.review_evidence_template_dir:
            templates = report.get("review_evidence_templates") or {}
            print(
                "wrote: "
                f"{args.review_evidence_template_dir} "
                f"({len(templates.get('files') or [])} review evidence templates)"
            )
        if args.review_packet_dir:
            packets = report.get("review_packets") or {}
            print(
                "wrote: "
                f"{args.review_packet_dir} "
                f"({len(packets.get('files') or [])} review packets)"
            )
        if args.next_review_batch_out:
            batch = report.get("next_review_batch") or {}
            print(
                "wrote: "
                f"{args.next_review_batch_out} "
                f"({batch.get('selected_count', 0)} review batch items)"
            )
        if args.review_kickoff_out:
            kickoff = report.get("review_kickoff") or {}
            print(
                "wrote: "
                f"{args.review_kickoff_out} "
                f"({kickoff.get('selected_count', 0)} review kickoff items)"
            )
        if args.source_coverage_plan_out:
            coverage_plan = report.get("source_coverage_plan") or {}
            print(
                "wrote: "
                f"{args.source_coverage_plan_out} "
                f"(coverage gaps={coverage_plan.get('coverage_gap_count', 0)})"
            )
        if args.review_bundle_out:
            bundle = report.get("review_bundle") or {}
            print(
                "wrote: "
                f"{args.review_bundle_out} "
                f"({bundle.get('review_packet_count', 0)} review packets, "
                f"{bundle.get('review_evidence_template_count', 0)} evidence templates)"
            )
        if args.apply_review_checklist_csv:
            checklist_apply = report.get("review_checklist_apply") or {}
            print(
                "REVIEW_CHECKLIST_APPLY: "
                f"ok={bool(checklist_apply.get('ok'))} "
                f"applied={checklist_apply.get('applied_count', 0)} "
                f"approved={checklist_apply.get('approved_count', 0)}"
            )
            if checklist_apply.get("updated_model_packs_out"):
                print(f"wrote: {checklist_apply['updated_model_packs_out']}")
        if args.validate_import_manifest:
            review = report.get("import_manifest_review") or {}
            print(
                "IMPORT_MANIFEST: "
                f"gate={'pass' if review.get('gate_passed') else 'blocked'} "
                f"included={review.get('included_count', 0)} "
                f"approved={review.get('approved_count', 0)}"
            )
        if args.validate_review_bundle:
            review = report.get("review_bundle_validation") or {}
            print(
                "REVIEW_BUNDLE: "
                f"ok={bool(review.get('ok'))} "
                f"artifacts={review.get('checked_artifact_count', 0)} "
                f"source_sha_match={bool(review.get('source_review_sha256_matches'))}"
            )
        if args.validate_next_review_batch:
            review = report.get("next_review_batch_validation") or {}
            print(
                "NEXT_REVIEW_BATCH: "
                f"ok={bool(review.get('ok'))} "
                f"artifacts={review.get('checked_artifact_count', 0)} "
                f"minimum_path={review.get('minimum_review_path_count', 0)} "
                f"source_sha_match={bool(review.get('source_review_sha256_matches'))}"
            )
        for blocker in report["blockers"][:10]:
            print(f"BLOCKED: {blocker}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"][:10]:
            print(f"WARN: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
