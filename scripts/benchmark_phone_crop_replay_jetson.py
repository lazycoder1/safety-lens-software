#!/usr/bin/env python3
"""Replay labeled phone stills through full-frame and person-crop pipelines.

The input directory uses a strict ``pos-*`` / ``neg-*`` filename convention.
Source paths, filenames, image bytes, detections, URLs, and credentials are
never written to the artifacts.  Each image becomes one paired accuracy sample
identified only by its SHA-256 digest.

The three output documents are accepted by ``video_eval.py paired-accuracy``::

    python scripts/video_eval.py paired-accuracy \
      --corpus OUT/corpus.json \
      --baseline OUT/baseline.json \
      --candidate OUT/candidate.json \
      --out OUT/paired-accuracy.json

This is an accuracy/latency replay, not a camera-capacity benchmark.  It uses
the production grouped-inference path and the configured model-server client,
but it does not exercise capture, tracking, alert confirmation, or scheduling.
Without explicit source grouping metadata, every still is conservatively placed
in one unverified bootstrap cluster; confidence intervals are therefore not
independent-cluster evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

RULE_NAME = "mobile_phone"
CAMERA_ID = "phone-crop-replay"
FORMAT_VERSION = "rakshak.phone_crop_replay.v1"
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
LABELED_STEM_PATTERN = re.compile(r"^(pos|neg)-[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126})$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEVICE_PATTERN = re.compile(r"^(?:cuda(?::[0-9]{1,2})?|cpu|mps|[0-9]{1,2})$")
DEFAULT_MAX_IMAGES = 10_000
DEFAULT_MAX_IMAGE_BYTES = 64 * 1024 * 1024
PHONE_CLASS_NAMES = frozenset({"cell phone", "mobile phone", "phone"})
KNOWN_INVOCATION_KEYS = (
    "coco_primary",
    "rtdetr_phone",
    "rtdetr_phone_fallback",
    "phone_person_crop",
)
KNOWN_CROP_PATHS = frozenset(
    {
        "person_crop",
        "full_frame",
        "full_frame_confirmation",
        "full_frame_shadow",
    }
)
ARTIFACT_FILENAMES = {
    "corpus": "corpus.json",
    "baseline": "baseline.json",
    "candidate": "candidate.json",
}


class ReplayInputError(ValueError):
    """Raised when labeled input cannot support an auditable comparison."""


class ReplayInferenceError(RuntimeError):
    """Raised when one inference result cannot be treated as a prediction."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _document_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one bounded regular file without following a final symlink."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReplayInputError("could not open one labeled image safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayInputError("labeled image inputs must be regular files")
        if metadata.st_size <= 0:
            raise ReplayInputError("labeled image inputs must not be empty")
        if metadata.st_size > maximum_bytes:
            raise ReplayInputError(
                "one labeled image exceeds the configured byte limit"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise ReplayInputError(
                "one labeled image exceeds the configured byte limit"
            )
        return payload
    finally:
        os.close(descriptor)


def _label_from_filename(path: Path) -> bool:
    match = LABELED_STEM_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ReplayInputError(
            "every supported image must use the strict pos-* or neg-* filename convention"
        )
    return match.group(1) == "pos"


def discover_labeled_images(
    image_directory: Path,
    *,
    maximum_images: int = DEFAULT_MAX_IMAGES,
    maximum_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> list[dict[str, Any]]:
    """Return deterministic private-path records for a strict labeled corpus."""

    if maximum_images < 2:
        raise ReplayInputError("maximum_images must be at least two")
    if maximum_image_bytes < 1:
        raise ReplayInputError("maximum_image_bytes must be positive")
    if image_directory.is_symlink() or not image_directory.is_dir():
        raise ReplayInputError("image_directory must be a real directory")
    root = image_directory.resolve()
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReplayInputError("symlinks are not allowed in the labeled image tree")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            paths.append(path)
    if not paths:
        raise ReplayInputError(
            "the labeled image directory contains no supported images"
        )
    if len(paths) > maximum_images:
        raise ReplayInputError(
            "the labeled image directory exceeds the image-count limit"
        )

    samples: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    positive_count = 0
    negative_count = 0
    for path in paths:
        truth = _label_from_filename(path)
        payload = _read_regular_file(path, maximum_bytes=maximum_image_bytes)
        digest = _sha256_bytes(payload)
        if digest in seen_digests:
            raise ReplayInputError(
                "duplicate image content is not allowed in a paired accuracy corpus"
            )
        seen_digests.add(digest)
        positive_count += int(truth)
        negative_count += int(not truth)
        samples.append(
            {
                # The path stays process-local and is removed by every artifact builder.
                "path": path,
                "source_sha256": digest,
                "truth": truth,
                "sample_id": f"img-{digest}",
                "timestamp_ms": 0,
            }
        )
    if positive_count == 0 or negative_count == 0:
        raise ReplayInputError(
            "filename-labeled replay requires at least one positive and one negative image"
        )
    ordered_samples = sorted(samples, key=lambda sample: sample["source_sha256"])
    cluster_id = f"unverified-cluster-{_source_set_sha256(ordered_samples)[:24]}"
    for sample in ordered_samples:
        # video_eval bootstraps by clip_id.  In the absence of trustworthy
        # capture/session grouping, one shared cluster is conservative and does
        # not manufacture independence from unrelated filenames.
        sample["clip_id"] = cluster_id
    return ordered_samples


def _source_set_sha256(samples: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(
        [
            {
                "source_sha256": sample["source_sha256"],
                "truth": sample["truth"],
            }
            for sample in samples
        ]
    )


def _script_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _git_source_provenance() -> dict[str, Any]:
    """Return hashes only; never serialize diff text or repository paths."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).stdout.strip()
        status_payload = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).stdout
        diff_payload = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {
            "gitAvailable": False,
            "gitCommit": None,
            "gitDirty": None,
            "sourceTreeStateSha256": None,
        }
    commit_text = commit.decode("ascii", errors="ignore").lower()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit_text):
        commit_text = ""
    return {
        "gitAvailable": bool(commit_text),
        "gitCommit": commit_text or None,
        "gitDirty": bool(status_payload),
        "sourceTreeStateSha256": _sha256_bytes(status_payload + b"\0" + diff_payload),
    }


def _capture_source_snapshot(*, require_clean: bool) -> dict[str, Any]:
    """Capture a stable script/tree identity without serializing source content."""

    script_sha256_before = _script_sha256()
    source_tree = _git_source_provenance()
    script_sha256_after = _script_sha256()
    if script_sha256_before != script_sha256_after:
        raise ReplayInputError(
            "the replay script changed while provenance was captured"
        )
    if source_tree.get("gitAvailable") is not True:
        raise ReplayInputError("an auditable replay requires an available Git worktree")
    if require_clean and source_tree.get("gitDirty") is not False:
        # This also rejects untracked source files, avoiding provenance that
        # identifies only their filenames rather than their executed bytes.
        raise ReplayInputError("an auditable replay requires a clean Git worktree")
    return {
        "scriptSha256": script_sha256_after,
        "sourceTree": source_tree,
    }


def build_corpus_document(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_set_sha256 = _source_set_sha256(samples)
    cluster_ids = {str(sample.get("clip_id") or "") for sample in samples}
    if len(cluster_ids) != 1 or "" in cluster_ids:
        raise ReplayInputError(
            "ungrouped still replay must use one conservative bootstrap cluster"
        )
    return {
        "schema_version": FORMAT_VERSION,
        "corpus_id": f"phone-crop-{source_set_sha256[:24]}",
        "rules": [RULE_NAME],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "clip_id": sample["clip_id"],
                "timestamp_ms": sample["timestamp_ms"],
                "source_sha256": sample["source_sha256"],
                "labels": {RULE_NAME: bool(sample["truth"])},
            }
            for sample in samples
        ],
        "events": [],
        "provenance": {
            "sourceSetSha256": source_set_sha256,
            "labelContract": "strict-lowercase-pos-neg-filename-prefix-v1",
            "labelContractSha256": _sha256_json(
                {
                    "positivePrefix": "pos-",
                    "negativePrefix": "neg-",
                    "rule": RULE_NAME,
                    "oneStillPerSourceFile": True,
                }
            ),
            "bootstrapClusterPolicy": "single-unverified-cluster-v1",
            "bootstrapClusterCount": 1,
            "bootstrapClusterVerified": False,
            "allSamplesShareUnverifiedBootstrapCluster": True,
            "oneStillPerBootstrapCluster": False,
            "nearDuplicateClusterReviewRequired": True,
            "containsSourcePaths": False,
            "containsFilenames": False,
            "containsImageBytes": False,
        },
    }


def build_inference_contract(
    *,
    confidence: float,
    phone_confidence: float,
    device: str,
    inference_size: int,
    maximum_input_dimension: int,
    person_crop_padding_fraction: float,
    person_crop_min_width: int,
    person_crop_min_height: int,
    person_crop_boundary_margin: int,
    person_crop_max_crops: int,
    person_crop_person_dedup_iou: float,
    person_crop_result_dedup_iou: float,
) -> dict[str, Any]:
    return {
        "cameraId": CAMERA_ID,
        "rule": RULE_NAME,
        "executionPlan": {
            "runCocoPrimary": True,
            "runRtdetrPhone": True,
            "frameBatchSizeHint": 1,
        },
        "comparison": {
            "baselinePhonePersonCropMode": "off",
            "candidatePhonePersonCropMode": "active",
            "pairedOrder": "alternating-per-sample",
        },
        "validityGate": {
            "minimumMeasuredTruthPositiveAuthoritativePersonCropSamples": 1,
            "minimumMeasuredTruthNegativeAuthoritativePersonCropSamples": 0,
            "authoritativePersonCropDefinition": (
                "path-person_crop-with-successful-measured-crop-invocation-v1"
            ),
        },
        "confidence": confidence,
        "phoneConfidence": phone_confidence,
        "device": device,
        "inferenceSize": inference_size,
        "maximumInputDimension": maximum_input_dimension,
        "inputDecode": "opencv-imdecode-bgr8",
        "inputResize": (
            "opencv-inter-area-preserve-aspect"
            if maximum_input_dimension > 0
            else "none"
        ),
        "personCropPolicy": {
            "paddingFraction": person_crop_padding_fraction,
            "minimumWidth": person_crop_min_width,
            "minimumHeight": person_crop_min_height,
            "boundaryMargin": person_crop_boundary_margin,
            "maximumCrops": person_crop_max_crops,
            "personDedupIou": person_crop_person_dedup_iou,
            "resultDedupIou": person_crop_result_dedup_iou,
        },
        "predictionPredicate": {
            "kind": "any-production-grouped-detection-class-v1",
            "acceptedClasses": sorted(PHONE_CLASS_NAMES),
        },
    }


def _execution_plan() -> dict[str, Any]:
    return {
        "capabilities": ["person_presence", "mobile_phone"],
        "required_model_keys": ["coco_primary"],
        "run_coco_primary": True,
        "run_rtdetr_phone": True,
        "run_ppe_specialist": False,
        "run_ppe_closed_set_candidate": False,
        "run_yoloe_long_tail": False,
        "run_fire_smoke_specialist": False,
        "run_pose_specialist": False,
        "run_face_recognition": False,
        "run_plate_recognition": False,
    }


def _runtime_config(contract: Mapping[str, Any], mode: str) -> dict[str, Any]:
    crop_policy = contract["personCropPolicy"]
    return {
        "global": {
            "phone_person_crop_mode": mode,
            "ppe_person_crop_mode": "off",
            "person_crop_padding_fraction": crop_policy["paddingFraction"],
            "person_crop_min_person_width": crop_policy["minimumWidth"],
            "person_crop_min_person_height": crop_policy["minimumHeight"],
            "person_crop_boundary_margin": crop_policy["boundaryMargin"],
            "person_crop_max_crops": crop_policy["maximumCrops"],
            "person_crop_person_dedup_iou": crop_policy["personDedupIou"],
            "person_crop_result_dedup_iou": crop_policy["resultDedupIou"],
        },
        "cameras": {
            CAMERA_ID: {
                "enabled": True,
                "safety_rule_ids": ["alert_mobile_phone"],
            }
        },
        "safety_rules": [
            {
                "id": "alert_mobile_phone",
                "name": "Mobile Phone Usage",
                "type": "alert",
                "classes": ["cell phone"],
                "severity": "P3",
                "enabled": True,
                "threshold": 1,
                "confidence": contract["phoneConfidence"],
            }
        ],
    }


@contextmanager
def _phone_crop_environment(mode: str):
    key = "SAFETYLENS_PHONE_PERSON_CROP_MODE"
    previous = os.environ.get(key)
    os.environ[key] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _decode_verified_sample(
    sample: Mapping[str, Any],
    *,
    maximum_bytes: int,
    maximum_dimension: int,
) -> np.ndarray:
    payload = _read_regular_file(Path(sample["path"]), maximum_bytes=maximum_bytes)
    if _sha256_bytes(payload) != sample["source_sha256"]:
        raise ReplayInputError("labeled image content changed after corpus discovery")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ReplayInputError("one labeled image could not be decoded as BGR8")
    if maximum_dimension > 0 and max(frame.shape[:2]) > maximum_dimension:
        scale = maximum_dimension / max(frame.shape[:2])
        frame = cv2.resize(
            frame,
            (
                max(1, math.floor(frame.shape[1] * scale)),
                max(1, math.floor(frame.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(frame)


def _bounded_counter(value: object) -> int:
    return value if type(value) is int and 0 <= value <= 1_000_000 else 0


def _invoke_variant(
    frame: np.ndarray,
    *,
    variant: str,
    contract: Mapping[str, Any],
    run_grouped_inference: Callable[..., object],
) -> dict[str, Any]:
    mode = "off" if variant == "baseline" else "active"
    cfg = _runtime_config(contract, mode)
    crop_telemetry: dict[str, Any] = {}
    started_ns = time.perf_counter_ns()
    try:
        with _phone_crop_environment(mode):
            result = run_grouped_inference(
                CAMERA_ID,
                frame.copy(),
                _execution_plan(),
                conf=float(contract["confidence"]),
                device=str(contract["device"]),
                imgsz=int(contract["inferenceSize"]),
                cfg=cfg,
                frame_batch_size_hint=1,
                person_crop_telemetry=crop_telemetry,
            )
    except Exception as exc:
        # Provider exceptions may include an endpoint or response body. Preserve
        # only a bounded exception type in benchmark output and logs.
        error_type = type(exc).__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type):
            error_type = "InferenceError"
        raise ReplayInferenceError(
            f"grouped inference failed with {error_type}"
        ) from None
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    if not isinstance(result, tuple) or len(result) != 4:
        raise ReplayInferenceError("grouped inference returned an invalid result shape")
    _annotated, detections, _pose, raw_invocations = result
    if not isinstance(detections, list) or not isinstance(raw_invocations, Mapping):
        raise ReplayInferenceError(
            "grouped inference returned invalid detections or counters"
        )
    invocations = {
        name: _bounded_counter(raw_invocations.get(name, 0))
        for name in KNOWN_INVOCATION_KEYS
    }
    if invocations["coco_primary"] < 1 or invocations["rtdetr_phone"] < 1:
        raise ReplayInferenceError(
            "the replay did not execute both required model paths"
        )
    if invocations["rtdetr_phone_fallback"]:
        # Treating transport/engine unavailability as a negative would silently
        # inflate false negatives and invalidate the crop comparison.
        raise ReplayInferenceError(
            "RT-DETR phone route fallback invalidated the replay"
        )

    prediction = any(
        isinstance(detection, Mapping)
        and str(detection.get("class") or "").strip().lower() in PHONE_CLASS_NAMES
        for detection in detections
    )
    phone_telemetry = crop_telemetry.get("phone")
    if variant == "candidate" and not isinstance(phone_telemetry, Mapping):
        raise ReplayInferenceError("active person-crop telemetry was not produced")
    phone_telemetry = phone_telemetry if isinstance(phone_telemetry, Mapping) else {}
    crop_attempts = _bounded_counter(phone_telemetry.get("cropInferenceAttempts", 0))
    raw_fallback_reasons = phone_telemetry.get("fallbackReasons")
    crop_failed = (
        isinstance(raw_fallback_reasons, (list, tuple))
        and "crop_inference_failed" in raw_fallback_reasons
    ) or (
        crop_attempts > 0 and phone_telemetry.get("cropInferenceSucceeded") is not True
    )
    if variant == "candidate" and crop_failed:
        # The production path deliberately fails open to full-frame inference.
        # That protects alerts, but it is not evidence for active crop accuracy
        # or latency and must invalidate this paired replay.
        raise ReplayInferenceError(
            "active person-crop inference failed before full-frame fallback"
        )
    raw_path = phone_telemetry.get("authoritativePath")
    authoritative_path = raw_path if raw_path in KNOWN_CROP_PATHS else None
    if variant == "candidate" and phone_telemetry.get("mode") != "active":
        raise ReplayInferenceError("candidate inference did not use active crop mode")
    if variant == "candidate" and authoritative_path not in {
        "person_crop",
        "full_frame",
    }:
        raise ReplayInferenceError(
            "candidate inference did not report an active-mode authoritative path"
        )
    if variant == "candidate" and crop_attempts > invocations["phone_person_crop"]:
        raise ReplayInferenceError(
            "candidate crop telemetry exceeded measured crop model invocations"
        )
    if (
        variant == "candidate"
        and authoritative_path == "person_crop"
        and (
            crop_attempts < 1
            or phone_telemetry.get("cropInferenceSucceeded") is not True
            or invocations["phone_person_crop"] < 1
        )
    ):
        raise ReplayInferenceError(
            "authoritative person-crop path lacked a successful measured crop invocation"
        )
    return {
        "prediction": bool(prediction),
        "inference_latency_ms": round(max(0.0, elapsed_ms), 6),
        "model_invocations": invocations,
        "person_crop": {
            "authoritativePath": authoritative_path,
            "cropInferenceAttempts": crop_attempts,
            "cropInferenceSucceeded": (
                phone_telemetry.get("cropInferenceSucceeded") is True
                if crop_attempts > 0
                else None
            ),
            "fullFrameInvocations": _bounded_counter(
                phone_telemetry.get("fullFrameInvocations", 0)
            ),
            "fallbackRequired": phone_telemetry.get("fallbackRequired") is True,
        },
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[rank], 6)


def _run_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(sample["inference_latency_ms"]) for sample in samples]
    invocation_totals: Counter[str] = Counter()
    path_totals: Counter[str] = Counter()
    fallback_count = 0
    for sample in samples:
        invocation_totals.update(sample["model_invocations"])
        path = sample["person_crop"].get("authoritativePath")
        if path:
            path_totals[str(path)] += 1
        fallback_count += int(sample["person_crop"].get("fallbackRequired") is True)
    return {
        "sampleCount": len(samples),
        "predictedPositiveCount": sum(
            int(sample["predictions"][RULE_NAME]) for sample in samples
        ),
        "latencyMs": {
            "minimum": round(min(latencies), 6) if latencies else None,
            "median": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "maximum": round(max(latencies), 6) if latencies else None,
        },
        "modelInvocationTotals": {
            key: invocation_totals[key] for key in KNOWN_INVOCATION_KEYS
        },
        "personCropAuthoritativePathCounts": dict(sorted(path_totals.items())),
        "personCropFallbackSampleCount": fallback_count,
    }


def _coverage_bucket(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_count = len(records)
    attempted_count = sum(
        int(_bounded_counter(record["person_crop"].get("cropInferenceAttempts")) > 0)
        for record in records
    )
    succeeded_count = sum(
        int(record["person_crop"].get("cropInferenceSucceeded") is True)
        for record in records
    )
    authoritative_count = sum(
        int(record["person_crop"].get("authoritativePath") == "person_crop")
        for record in records
    )
    full_frame_count = sum(
        int(
            record["person_crop"].get("authoritativePath")
            in {"full_frame", "full_frame_confirmation", "full_frame_shadow"}
        )
        for record in records
    )

    def rate(count: int) -> float | None:
        return round(count / sample_count, 6) if sample_count else None

    return {
        "labeledSampleCount": sample_count,
        "cropAttemptedSampleCount": attempted_count,
        "cropAttemptedRate": rate(attempted_count),
        "cropInferenceSucceededSampleCount": succeeded_count,
        "cropInferenceSucceededRate": rate(succeeded_count),
        "authoritativePersonCropSampleCount": authoritative_count,
        "authoritativePersonCropRate": rate(authoritative_count),
        "fullFrameAuthoritativeSampleCount": full_frame_count,
        "fullFrameAuthoritativeRate": rate(full_frame_count),
    }


def build_candidate_person_crop_coverage(
    labeled_samples: Sequence[Mapping[str, Any]],
    candidate_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate measured crop-path coverage and enforce the publication gate."""

    truth_by_sample: dict[str, bool] = {}
    for sample in labeled_samples:
        sample_id = str(sample.get("sample_id") or "")
        truth = sample.get("truth")
        if not sample_id or sample_id in truth_by_sample or type(truth) is not bool:
            raise ReplayInputError("labeled samples have invalid coverage identities")
        truth_by_sample[sample_id] = truth

    candidate_by_sample: dict[str, Mapping[str, Any]] = {}
    for sample in candidate_samples:
        sample_id = str(sample.get("sample_id") or "")
        crop = sample.get("person_crop")
        if (
            not sample_id
            or sample_id in candidate_by_sample
            or not isinstance(crop, Mapping)
        ):
            raise ReplayInferenceError(
                "candidate samples have invalid crop coverage identities"
            )
        candidate_by_sample[sample_id] = sample
    if set(candidate_by_sample) != set(truth_by_sample):
        raise ReplayInferenceError(
            "candidate crop coverage does not exactly match the labeled corpus"
        )

    positive_records: list[Mapping[str, Any]] = []
    negative_records: list[Mapping[str, Any]] = []
    for sample_id, truth in truth_by_sample.items():
        record = candidate_by_sample[sample_id]
        crop = record["person_crop"]
        attempts = _bounded_counter(crop.get("cropInferenceAttempts"))
        authoritative = crop.get("authoritativePath") == "person_crop"
        if crop.get("cropInferenceSucceeded") is True and attempts < 1:
            raise ReplayInferenceError(
                "crop coverage reported success without a measured crop attempt"
            )
        if authoritative and (
            attempts < 1 or crop.get("cropInferenceSucceeded") is not True
        ):
            raise ReplayInferenceError(
                "authoritative crop coverage lacked a successful measured crop"
            )
        (positive_records if truth else negative_records).append(record)

    overall = _coverage_bucket(list(candidate_by_sample.values()))
    positive = _coverage_bucket(positive_records)
    negative = _coverage_bucket(negative_records)
    gate_passed = positive["authoritativePersonCropSampleCount"] >= 1
    coverage = {
        "gate": {
            "name": "measured-truth-positive-authoritative-person-crop-v1",
            "minimumTruthPositiveAuthoritativePersonCropSamples": 1,
            # A correct negative crop normally has no phone candidate and the
            # production policy delegates authority to the full-frame path.
            "minimumTruthNegativeAuthoritativePersonCropSamples": 0,
            "passed": gate_passed,
        },
        "overall": overall,
        "byGroundTruth": {
            "positive": positive,
            "negative": negative,
        },
    }
    if not gate_passed:
        raise ReplayInferenceError(
            "candidate produced no measured truth-positive authoritative person-crop sample"
        )
    return coverage


def execute_paired_replay(
    samples: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    run_grouped_inference: Callable[..., object],
    warmups: int = 1,
    maximum_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> dict[str, list[dict[str, Any]]]:
    """Execute each measured image once per variant with alternating order."""

    if not samples:
        raise ReplayInputError("the replay has no samples")
    if warmups < 0 or warmups > 100:
        raise ReplayInputError("warmups must be between zero and 100")
    maximum_dimension = int(contract["maximumInputDimension"])
    # Validate the entire immutable corpus before the first model call.  A
    # corrupt or replaced late-sorted image must not leave a partially executed
    # comparison that an operator could mistake for complete evidence.
    for sample in samples:
        _decode_verified_sample(
            sample,
            maximum_bytes=maximum_image_bytes,
            maximum_dimension=maximum_dimension,
        )
    if warmups:
        warmup_frame = _decode_verified_sample(
            samples[0],
            maximum_bytes=maximum_image_bytes,
            maximum_dimension=maximum_dimension,
        )
        for warmup_index in range(warmups):
            order = (
                ("baseline", "candidate")
                if warmup_index % 2 == 0
                else ("candidate", "baseline")
            )
            for variant in order:
                _invoke_variant(
                    warmup_frame,
                    variant=variant,
                    contract=contract,
                    run_grouped_inference=run_grouped_inference,
                )

    results: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "candidate": [],
    }
    for index, sample in enumerate(samples):
        frame = _decode_verified_sample(
            sample,
            maximum_bytes=maximum_image_bytes,
            maximum_dimension=maximum_dimension,
        )
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for variant in order:
            outcome = _invoke_variant(
                frame,
                variant=variant,
                contract=contract,
                run_grouped_inference=run_grouped_inference,
            )
            results[variant].append(
                {
                    "sample_id": sample["sample_id"],
                    "clip_id": sample["clip_id"],
                    "timestamp_ms": sample["timestamp_ms"],
                    "source_sha256": sample["source_sha256"],
                    "predictions": {RULE_NAME: outcome["prediction"]},
                    "inference_latency_ms": outcome["inference_latency_ms"],
                    "model_invocations": outcome["model_invocations"],
                    "person_crop": outcome["person_crop"],
                }
            )
    for variant_samples in results.values():
        variant_samples.sort(key=lambda sample: sample["sample_id"])
    return results


def build_run_document(
    variant: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    corpus_document: Mapping[str, Any],
    corpus_document_sha256: str,
    inference_contract: Mapping[str, Any],
    warmups: int,
    model_artifact_sha256: Mapping[str, str],
    generated_at: str,
    source_provenance: Mapping[str, Any],
    script_sha256: str,
    run_set_id: str,
    measured_person_crop_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if variant not in {"baseline", "candidate"}:
        raise ReplayInputError("run variant must be baseline or candidate")
    performance = _run_summary(samples)
    if variant == "candidate":
        if (
            not isinstance(measured_person_crop_coverage, Mapping)
            or not isinstance(measured_person_crop_coverage.get("gate"), Mapping)
            or measured_person_crop_coverage["gate"].get("passed") is not True
        ):
            raise ReplayInferenceError(
                "candidate document requires passing measured person-crop coverage"
            )
        performance["measuredPersonCropCoverage"] = dict(measured_person_crop_coverage)
    elif measured_person_crop_coverage is not None:
        raise ReplayInputError(
            "baseline document cannot include candidate crop coverage"
        )
    contract_sha256 = _sha256_json(inference_contract)
    return {
        "schema_version": FORMAT_VERSION,
        "run_id": f"{corpus_document['corpus_id']}-{variant}-{run_set_id[-16:]}",
        "corpus_id": corpus_document["corpus_id"],
        "variant": (
            "full_frame_baseline" if variant == "baseline" else "active_person_crop"
        ),
        "samples": list(samples),
        "alerts": [],
        "performance": performance,
        "provenance": {
            "generatedAt": generated_at,
            "runSetId": run_set_id,
            "corpusDocumentSha256": corpus_document_sha256,
            "sourceSetSha256": corpus_document["provenance"]["sourceSetSha256"],
            "scriptSha256": script_sha256,
            "inferenceContractSha256": contract_sha256,
            "inferenceContract": inference_contract,
            "modelArtifactSha256": dict(sorted(model_artifact_sha256.items())),
            "modelArtifactIdentitySource": "operator_asserted_not_endpoint_attested",
            "sourceTree": dict(source_provenance),
            "modelTransport": "configured_model_server_client",
            "containsSourcePaths": False,
            "containsFilenames": False,
            "containsImageBytes": False,
            "containsDetections": False,
            "containsUrls": False,
            "warmupInferenceCountPerVariant": warmups,
            "measuredInferenceCount": len(samples),
        },
    }


def build_run_set_id(
    *,
    source_set_sha256: str,
    inference_contract: Mapping[str, Any],
    model_artifact_sha256: Mapping[str, str],
    generated_at: str,
    source_provenance: Mapping[str, Any],
    script_sha256: str,
) -> str:
    digest = _sha256_json(
        {
            "sourceSetSha256": source_set_sha256,
            "inferenceContractSha256": _sha256_json(inference_contract),
            "modelArtifactSha256": dict(sorted(model_artifact_sha256.items())),
            "generatedAt": generated_at,
            "scriptSha256": script_sha256,
            "sourceTree": dict(source_provenance),
        }
    )
    return f"phone-crop-run-{digest[:32]}"


def _validated_sha256(value: object, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ReplayInputError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _validated_device(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if not DEVICE_PATTERN.fullmatch(normalized):
        raise ReplayInputError(
            "device must be cpu, mps, cuda, cuda:N, or a numeric CUDA index"
        )
    return normalized


def _atomic_write_document(path: Path, document: Mapping[str, Any]) -> str:
    payload = _document_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return _sha256_bytes(payload)


def _atomic_write_artifact_set(
    output_directory: Path,
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Publish one complete three-document generation with one directory rename."""

    if set(documents) != set(ARTIFACT_FILENAMES):
        raise ReplayInputError(
            "artifact set must contain corpus, baseline, and candidate"
        )
    if os.path.lexists(output_directory):
        raise ReplayInputError("out-dir must not already exist")
    parent = output_directory.parent
    staging: Path | None = None
    published = False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".phone-crop-replay.",
                suffix=".tmp",
                dir=parent,
            )
        )
        hashes = {
            name: _atomic_write_document(
                staging / ARTIFACT_FILENAMES[name],
                documents[name],
            )
            for name in ("corpus", "baseline", "candidate")
        }
        directory_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if os.path.lexists(output_directory):
            raise ReplayInputError("out-dir appeared while artifacts were staged")
        os.rename(staging, output_directory)
        published = True
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return hashes
    except ReplayInputError:
        raise
    except OSError:
        raise ReplayInputError("could not publish the complete artifact set") from None
    finally:
        if staging is not None and not published:
            shutil.rmtree(staging, ignore_errors=True)


def _finite_float(value: object, label: str, minimum: float, maximum: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayInputError(f"{label} must be numeric") from exc
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ReplayInputError(f"{label} must be between {minimum} and {maximum}")
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="new directory atomically created for this complete artifact set",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--phone-conf", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-input-dimension", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--maximum-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument(
        "--maximum-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES
    )
    parser.add_argument("--person-crop-padding-fraction", type=float, default=0.12)
    parser.add_argument("--person-crop-min-width", type=int, default=24)
    parser.add_argument("--person-crop-min-height", type=int, default=48)
    parser.add_argument("--person-crop-boundary-margin", type=int, default=2)
    parser.add_argument("--person-crop-max-crops", type=int, default=8)
    parser.add_argument("--person-crop-person-dedup-iou", type=float, default=0.85)
    parser.add_argument("--person-crop-result-dedup-iou", type=float, default=0.55)
    parser.add_argument(
        "--coco-model-sha256",
        default=os.environ.get("SAFETYLENS_COCO_MODEL_SHA256", ""),
        help="operator-attested SHA-256 of the COCO artifact used by model-server",
    )
    parser.add_argument(
        "--phone-model-sha256",
        default=os.environ.get("SAFETYLENS_RTDETR_PHONE_BATCH1_ENGINE_SHA256", ""),
        help="operator-attested SHA-256 of the RT-DETR phone batch-1 engine",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        source_snapshot = _capture_source_snapshot(require_clean=True)
        confidence = _finite_float(args.conf, "conf", 0.0, 1.0)
        phone_confidence = _finite_float(args.phone_conf, "phone-conf", 0.0, 1.0)
        if not 160 <= args.imgsz <= 1920:
            raise ReplayInputError("imgsz must be between 160 and 1920")
        if not 0 <= args.max_input_dimension <= 16_384:
            raise ReplayInputError("max-input-dimension must be zero or at most 16384")
        if not 0 <= args.warmups <= 100:
            raise ReplayInputError("warmups must be between zero and 100")
        if not 1 <= args.person_crop_min_width <= 4096:
            raise ReplayInputError("person-crop-min-width is out of range")
        if not 1 <= args.person_crop_min_height <= 4096:
            raise ReplayInputError("person-crop-min-height is out of range")
        if not 0 <= args.person_crop_boundary_margin <= 256:
            raise ReplayInputError("person-crop-boundary-margin is out of range")
        if not 1 <= args.person_crop_max_crops <= 32:
            raise ReplayInputError("person-crop-max-crops is out of range")
        padding_fraction = _finite_float(
            args.person_crop_padding_fraction,
            "person-crop-padding-fraction",
            0.0,
            1.0,
        )
        person_dedup_iou = _finite_float(
            args.person_crop_person_dedup_iou,
            "person-crop-person-dedup-iou",
            0.0,
            1.0,
        )
        result_dedup_iou = _finite_float(
            args.person_crop_result_dedup_iou,
            "person-crop-result-dedup-iou",
            0.0,
            1.0,
        )
        device = _validated_device(args.device)
        model_artifact_sha256 = {
            "coco_primary": _validated_sha256(
                args.coco_model_sha256, "coco-model-sha256"
            ),
            "rtdetr_phone_batch1": _validated_sha256(
                args.phone_model_sha256, "phone-model-sha256"
            ),
        }
        if os.path.lexists(args.out_dir):
            raise ReplayInputError("out-dir must not already exist")
        samples = discover_labeled_images(
            args.images,
            maximum_images=args.maximum_images,
            maximum_image_bytes=args.maximum_image_bytes,
        )
        contract = build_inference_contract(
            confidence=confidence,
            phone_confidence=phone_confidence,
            device=device,
            inference_size=args.imgsz,
            maximum_input_dimension=args.max_input_dimension,
            person_crop_padding_fraction=padding_fraction,
            person_crop_min_width=args.person_crop_min_width,
            person_crop_min_height=args.person_crop_min_height,
            person_crop_boundary_margin=args.person_crop_boundary_margin,
            person_crop_max_crops=args.person_crop_max_crops,
            person_crop_person_dedup_iou=person_dedup_iou,
            person_crop_result_dedup_iou=result_dedup_iou,
        )

        import model_manager
        import video_processing

        try:
            remote_enabled = model_manager.is_remote_inference_enabled()
            remote_healthy = (
                model_manager.warm_remote_inference_session()
                if remote_enabled
                else False
            )
            readiness = (
                model_manager.model_readiness_snapshot(["coco_primary"])
                if remote_healthy
                else {}
            )
        except Exception:
            raise ReplayInferenceError(
                "the configured model-server readiness check failed"
            ) from None
        if not remote_enabled:
            raise ReplayInferenceError(
                "the configured model-server client is not enabled"
            )
        if not remote_healthy:
            raise ReplayInferenceError(
                "the configured model-server health check failed"
            )
        if readiness.get("coco_primary") is not True:
            raise ReplayInferenceError("the COCO primary route is not ready")

        replay_results = execute_paired_replay(
            samples,
            contract=contract,
            run_grouped_inference=video_processing._run_grouped_inference,
            warmups=args.warmups,
            maximum_image_bytes=args.maximum_image_bytes,
        )
        measured_person_crop_coverage = build_candidate_person_crop_coverage(
            samples,
            replay_results["candidate"],
        )
        if _capture_source_snapshot(require_clean=False) != source_snapshot:
            raise ReplayInferenceError(
                "the source tree changed while the replay was running"
            )
        corpus = build_corpus_document(samples)
        generated_at = datetime.now(timezone.utc).isoformat()
        source_provenance = source_snapshot["sourceTree"]
        script_sha256 = source_snapshot["scriptSha256"]
        run_set_id = build_run_set_id(
            source_set_sha256=corpus["provenance"]["sourceSetSha256"],
            inference_contract=contract,
            model_artifact_sha256=model_artifact_sha256,
            generated_at=generated_at,
            source_provenance=source_provenance,
            script_sha256=script_sha256,
        )
        corpus["provenance"]["runSetId"] = run_set_id
        corpus_document_sha256 = _sha256_bytes(_document_bytes(corpus))
        baseline = build_run_document(
            "baseline",
            replay_results["baseline"],
            corpus_document=corpus,
            corpus_document_sha256=corpus_document_sha256,
            inference_contract=contract,
            warmups=args.warmups,
            model_artifact_sha256=model_artifact_sha256,
            generated_at=generated_at,
            source_provenance=source_provenance,
            script_sha256=script_sha256,
            run_set_id=run_set_id,
        )
        candidate = build_run_document(
            "candidate",
            replay_results["candidate"],
            corpus_document=corpus,
            corpus_document_sha256=corpus_document_sha256,
            inference_contract=contract,
            warmups=args.warmups,
            model_artifact_sha256=model_artifact_sha256,
            generated_at=generated_at,
            source_provenance=source_provenance,
            script_sha256=script_sha256,
            run_set_id=run_set_id,
            measured_person_crop_coverage=measured_person_crop_coverage,
        )

        # Nothing is published until all three documents have been constructed.
        document_hashes = _atomic_write_artifact_set(
            args.out_dir,
            {
                "corpus": corpus,
                "baseline": baseline,
                "candidate": candidate,
            },
        )
    except (ReplayInputError, ReplayInferenceError) as exc:
        parser.exit(2, f"phone-crop replay failed: {exc}\n")

    print(
        json.dumps(
            {
                "ok": True,
                "sampleCount": len(samples),
                "positiveCount": sum(int(sample["truth"]) for sample in samples),
                "negativeCount": sum(int(not sample["truth"]) for sample in samples),
                "sourceSetSha256": corpus["provenance"]["sourceSetSha256"],
                "documentSha256": document_hashes,
                "containsSensitivePayloads": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
