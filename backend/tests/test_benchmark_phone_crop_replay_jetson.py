"""Focused tests for the aggregate-only phone person-crop replay harness."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark_phone_crop_replay_jetson.py"
VIDEO_EVAL = ROOT / "scripts" / "video_eval.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def replay():
    return _load_module("benchmark_phone_crop_replay_test", SCRIPT)


@pytest.fixture(scope="module")
def video_eval():
    return _load_module("video_eval_phone_crop_replay_test", VIDEO_EVAL)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((96, 128, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)


def _contract(replay, *, maximum_dimension: int = 0):
    return replay.build_inference_contract(
        confidence=0.35,
        phone_confidence=0.15,
        device="cuda",
        inference_size=640,
        maximum_input_dimension=maximum_dimension,
        person_crop_padding_fraction=0.12,
        person_crop_min_width=24,
        person_crop_min_height=48,
        person_crop_boundary_margin=2,
        person_crop_max_crops=8,
        person_crop_person_dedup_iou=0.85,
        person_crop_result_dedup_iou=0.55,
    )


def _model_hashes():
    return {"coco_primary": "c" * 64, "rtdetr_phone_batch1": "d" * 64}


def _source_provenance():
    return {
        "gitAvailable": True,
        "gitCommit": "a" * 40,
        "gitDirty": False,
        "sourceTreeStateSha256": "b" * 64,
    }


def _script_hash():
    return "e" * 64


def test_discovery_is_content_addressed_deterministic_and_path_free(
    replay, tmp_path: Path
):
    _write_image(tmp_path / "nested" / "pos-worker.PNG", 220)
    _write_image(tmp_path / "neg-empty.jpg", 10)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    first = replay.discover_labeled_images(tmp_path)
    second = replay.discover_labeled_images(tmp_path)
    assert [sample["source_sha256"] for sample in first] == sorted(
        sample["source_sha256"] for sample in first
    )
    assert [sample["source_sha256"] for sample in first] == [
        sample["source_sha256"] for sample in second
    ]
    assert {sample["truth"] for sample in first} == {True, False}
    assert all(
        sample["sample_id"] == f"img-{sample['source_sha256']}" for sample in first
    )
    assert len({sample["clip_id"] for sample in first}) == 1
    assert first[0]["clip_id"].startswith("unverified-cluster-")

    corpus = replay.build_corpus_document(first)
    rendered = json.dumps(corpus, sort_keys=True)
    assert "pos-worker" not in rendered
    assert "neg-empty" not in rendered
    assert str(tmp_path) not in rendered
    assert corpus["provenance"]["containsFilenames"] is False
    assert corpus["provenance"]["containsSourcePaths"] is False
    assert corpus["provenance"]["bootstrapClusterCount"] == 1
    assert corpus["provenance"]["bootstrapClusterVerified"] is False
    assert corpus["provenance"]["oneStillPerBootstrapCluster"] is False
    assert all(
        replay.SHA256_PATTERN.fullmatch(sample["source_sha256"])
        for sample in corpus["samples"]
    )


def test_ungrouped_stills_use_one_unverified_bootstrap_cluster(
    replay, video_eval, tmp_path: Path
):
    _write_image(tmp_path / "pos-one.jpg", 240)
    _write_image(tmp_path / "pos-two.jpg", 180)
    _write_image(tmp_path / "neg-one.jpg", 20)
    samples = replay.discover_labeled_images(tmp_path)

    corpus = replay.build_corpus_document(samples)
    normalized = video_eval.validate_paired_accuracy_corpus(corpus)

    assert len(normalized["clip_ids"]) == 1
    assert {sample["clip_id"] for sample in corpus["samples"]} == set(
        normalized["clip_ids"]
    )
    assert corpus["provenance"]["bootstrapClusterPolicy"] == (
        "single-unverified-cluster-v1"
    )
    assert corpus["provenance"]["bootstrapClusterVerified"] is False
    assert corpus["provenance"]["nearDuplicateClusterReviewRequired"] is True

    invalid = [dict(sample) for sample in samples]
    invalid[0]["clip_id"] = "falsely-independent-image"
    with pytest.raises(replay.ReplayInputError, match="one conservative"):
        replay.build_corpus_document(invalid)


@pytest.mark.parametrize("bad_name", ["positive-x.jpg", "POS-x.jpg", "pos-.jpg"])
def test_supported_images_with_noncanonical_labels_are_fatal(
    replay, tmp_path: Path, bad_name: str
):
    _write_image(tmp_path / bad_name, 200)
    _write_image(tmp_path / "neg-valid.jpg", 0)

    with pytest.raises(replay.ReplayInputError, match=r"strict pos-\* or neg-\*"):
        replay.discover_labeled_images(tmp_path)


def test_duplicate_bytes_and_single_class_corpora_are_rejected(replay, tmp_path: Path):
    _write_image(tmp_path / "pos-one.jpg", 100)
    (tmp_path / "neg-copy.jpg").write_bytes((tmp_path / "pos-one.jpg").read_bytes())

    with pytest.raises(replay.ReplayInputError, match="duplicate image content"):
        replay.discover_labeled_images(tmp_path)

    (tmp_path / "neg-copy.jpg").unlink()
    with pytest.raises(replay.ReplayInputError, match="at least one positive"):
        replay.discover_labeled_images(tmp_path)


def test_symlinks_are_rejected(replay, tmp_path: Path):
    _write_image(tmp_path / "pos-one.jpg", 180)
    _write_image(tmp_path / "neg-one.jpg", 0)
    try:
        (tmp_path / "linked.jpg").symlink_to(tmp_path / "pos-one.jpg")
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(replay.ReplayInputError, match="symlinks"):
        replay.discover_labeled_images(tmp_path)


def test_replay_outputs_pass_paired_validator_without_sensitive_payloads(
    replay, video_eval, tmp_path: Path, monkeypatch
):
    _write_image(tmp_path / "pos-phone.jpg", 255)
    _write_image(tmp_path / "neg-clear.jpg", 0)
    samples = replay.discover_labeled_images(tmp_path)
    modes: list[str] = []

    def grouped(_camera_id, frame, _plan, *, cfg, person_crop_telemetry, **_options):
        mode = cfg["global"]["phone_person_crop_mode"]
        modes.append(mode)
        positive_image = float(frame.mean()) > 100.0
        prediction = True if mode == "off" else positive_image
        if mode == "active":
            person_crop_telemetry["phone"] = {
                "mode": "active",
                "authoritativePath": "person_crop" if positive_image else "full_frame",
                "cropInferenceAttempts": int(positive_image),
                "cropInferenceSucceeded": bool(positive_image),
                "fullFrameInvocations": int(not positive_image),
                "fallbackRequired": not positive_image,
            }
        detections = [{"class": "cell phone"}] if prediction else []
        return (
            None,
            detections,
            None,
            {
                "coco_primary": 1,
                "rtdetr_phone": 1,
                "rtdetr_phone_fallback": 0,
                "phone_person_crop": int(mode == "active" and positive_image),
            },
        )

    monkeypatch.setenv("SAFETYLENS_PHONE_PERSON_CROP_MODE", "shadow")
    contract = _contract(replay, maximum_dimension=64)
    results = replay.execute_paired_replay(
        samples,
        contract=contract,
        run_grouped_inference=grouped,
        warmups=1,
    )
    assert os.environ["SAFETYLENS_PHONE_PERSON_CROP_MODE"] == "shadow"
    assert modes.count("off") == 3
    assert modes.count("active") == 3
    measured_coverage = replay.build_candidate_person_crop_coverage(
        samples,
        results["candidate"],
    )

    corpus_document = replay.build_corpus_document(samples)
    generated_at = "2026-07-15T00:00:00+00:00"
    run_set_id = replay.build_run_set_id(
        source_set_sha256=corpus_document["provenance"]["sourceSetSha256"],
        inference_contract=contract,
        model_artifact_sha256=_model_hashes(),
        generated_at=generated_at,
        source_provenance=_source_provenance(),
        script_sha256=_script_hash(),
    )
    corpus_document["provenance"]["runSetId"] = run_set_id
    corpus_sha = replay._sha256_bytes(replay._document_bytes(corpus_document))
    baseline_document = replay.build_run_document(
        "baseline",
        results["baseline"],
        corpus_document=corpus_document,
        corpus_document_sha256=corpus_sha,
        inference_contract=contract,
        warmups=1,
        model_artifact_sha256=_model_hashes(),
        generated_at=generated_at,
        source_provenance=_source_provenance(),
        script_sha256=_script_hash(),
        run_set_id=run_set_id,
    )
    candidate_document = replay.build_run_document(
        "candidate",
        results["candidate"],
        corpus_document=corpus_document,
        corpus_document_sha256=corpus_sha,
        inference_contract=contract,
        warmups=1,
        model_artifact_sha256=_model_hashes(),
        generated_at=generated_at,
        source_provenance=_source_provenance(),
        script_sha256=_script_hash(),
        run_set_id=run_set_id,
        measured_person_crop_coverage=measured_coverage,
    )

    normalized_corpus = video_eval.validate_paired_accuracy_corpus(corpus_document)
    normalized_baseline = video_eval.validate_paired_accuracy_run(
        baseline_document, normalized_corpus, "baseline"
    )
    normalized_candidate = video_eval.validate_paired_accuracy_run(
        candidate_document, normalized_corpus, "candidate"
    )
    report = video_eval.build_paired_accuracy_report(
        normalized_corpus,
        normalized_baseline,
        normalized_candidate,
        corpus_sha256=corpus_sha,
        baseline_sha256="1" * 64,
        candidate_sha256="2" * 64,
        bootstrap_resamples=100,
        generated_at="2026-07-15T00:00:00+00:00",
    )
    metrics = report["frame_rule_metrics"][replay.RULE_NAME]
    assert metrics["baseline"]["true_positive"] == 1
    assert metrics["baseline"]["false_positive"] == 1
    assert metrics["candidate"]["precision"] == 1.0
    assert metrics["candidate"]["recall"] == 1.0
    assert report["methodology"]["cluster_count"] == 1
    assert report["methodology"]["single_cluster_interval_warning"] is True
    assert baseline_document["run_id"] != candidate_document["run_id"]
    assert all(
        type(sample["predictions"][replay.RULE_NAME]) is bool
        and sample["inference_latency_ms"] >= 0
        for sample in candidate_document["samples"]
    )

    rendered = json.dumps(
        [corpus_document, baseline_document, candidate_document], sort_keys=True
    )
    assert "pos-phone" not in rendered
    assert "neg-clear" not in rendered
    assert str(tmp_path) not in rendered
    assert "http://" not in rendered
    assert "https://" not in rendered
    assert "bbox" not in rendered
    assert candidate_document["provenance"]["containsDetections"] is False
    assert candidate_document["performance"]["sampleCount"] == 2
    coverage = candidate_document["performance"]["measuredPersonCropCoverage"]
    assert coverage["gate"]["passed"] is True
    assert (
        coverage["byGroundTruth"]["positive"]["authoritativePersonCropSampleCount"] == 1
    )
    assert (
        coverage["byGroundTruth"]["negative"]["authoritativePersonCropSampleCount"] == 0
    )
    assert coverage["gate"]["minimumTruthNegativeAuthoritativePersonCropSamples"] == 0
    assert corpus_document["provenance"]["runSetId"] == run_set_id
    assert baseline_document["provenance"]["runSetId"] == run_set_id
    assert candidate_document["provenance"]["runSetId"] == run_set_id
    assert baseline_document["provenance"]["corpusDocumentSha256"] == corpus_sha
    assert candidate_document["provenance"]["corpusDocumentSha256"] == corpus_sha
    assert baseline_document["provenance"]["scriptSha256"] == _script_hash()
    assert candidate_document["provenance"]["scriptSha256"] == _script_hash()


def test_all_full_frame_candidate_is_rejected_by_prepublication_crop_gate(
    replay, tmp_path: Path
):
    _write_image(tmp_path / "pos-phone.jpg", 255)
    _write_image(tmp_path / "neg-clear.jpg", 0)
    samples = replay.discover_labeled_images(tmp_path)

    def grouped(_camera_id, frame, _plan, *, cfg, person_crop_telemetry, **_options):
        mode = cfg["global"]["phone_person_crop_mode"]
        if mode == "active":
            # Crops ran successfully, but none became authoritative. This still
            # cannot validate the candidate's crop-only accuracy or latency.
            person_crop_telemetry["phone"] = {
                "mode": "active",
                "authoritativePath": "full_frame",
                "cropInferenceAttempts": 1,
                "cropInferenceSucceeded": True,
                "fullFrameInvocations": 1,
                "fallbackRequired": False,
            }
        detections = [{"class": "cell phone"}] if float(frame.mean()) > 100 else []
        return (
            None,
            detections,
            None,
            {
                "coco_primary": 1,
                "rtdetr_phone": 1,
                "rtdetr_phone_fallback": 0,
                "phone_person_crop": int(mode == "active"),
            },
        )

    results = replay.execute_paired_replay(
        samples,
        contract=_contract(replay),
        run_grouped_inference=grouped,
        warmups=0,
    )

    with pytest.raises(
        replay.ReplayInferenceError,
        match="no measured truth-positive authoritative person-crop",
    ):
        replay.build_candidate_person_crop_coverage(
            samples,
            results["candidate"],
        )


def test_rtdetr_fallback_aborts_instead_of_becoming_a_negative(replay):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def grouped(*_args, **_kwargs):
        return (
            None,
            [],
            None,
            {
                "coco_primary": 1,
                "rtdetr_phone": 1,
                "rtdetr_phone_fallback": 1,
            },
        )

    with pytest.raises(replay.ReplayInferenceError, match="fallback invalidated"):
        replay._invoke_variant(
            frame,
            variant="baseline",
            contract=_contract(replay),
            run_grouped_inference=grouped,
        )


def test_provider_exception_details_are_not_exposed(replay, monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def grouped(*_args, **_kwargs):
        raise RuntimeError("https://user:secret@private.example/model")

    monkeypatch.setenv("SAFETYLENS_PHONE_PERSON_CROP_MODE", "shadow")
    with pytest.raises(
        replay.ReplayInferenceError,
        match="grouped inference failed with RuntimeError",
    ) as caught:
        replay._invoke_variant(
            frame,
            variant="baseline",
            contract=_contract(replay),
            run_grouped_inference=grouped,
        )
    assert "private.example" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert os.environ["SAFETYLENS_PHONE_PERSON_CROP_MODE"] == "shadow"


def test_caught_crop_failure_invalidates_candidate_even_when_full_frame_succeeds(
    replay,
):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def grouped(*_args, person_crop_telemetry, **_kwargs):
        person_crop_telemetry["phone"] = {
            "mode": "active",
            "authoritativePath": "full_frame",
            "cropInferenceAttempts": 1,
            "cropInferenceSucceeded": False,
            "fullFrameInvocations": 1,
            "fallbackRequired": True,
            "fallbackReasons": ["crop_inference_failed"],
        }
        return (
            None,
            [{"class": "cell phone"}],
            None,
            {
                "coco_primary": 1,
                "rtdetr_phone": 2,
                "rtdetr_phone_fallback": 0,
                "phone_person_crop": 1,
            },
        )

    with pytest.raises(
        replay.ReplayInferenceError,
        match="active person-crop inference failed",
    ):
        replay._invoke_variant(
            frame,
            variant="candidate",
            contract=_contract(replay),
            run_grouped_inference=grouped,
        )


def test_source_mutation_after_discovery_aborts_before_inference(
    replay, tmp_path: Path
):
    positive = tmp_path / "pos-phone.jpg"
    _write_image(positive, 255)
    _write_image(tmp_path / "neg-clear.jpg", 0)
    samples = replay.discover_labeled_images(tmp_path)
    _write_image(positive, 120)
    calls = 0

    def grouped(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    with pytest.raises(replay.ReplayInputError, match="changed after corpus discovery"):
        replay.execute_paired_replay(
            samples,
            contract=_contract(replay),
            run_grouped_inference=grouped,
            warmups=0,
        )
    assert calls == 0


def test_atomic_output_is_private_and_leaves_no_temporary_file(replay, tmp_path: Path):
    path = tmp_path / "out" / "corpus.json"
    document = {"corpus_id": "fixture", "samples": []}

    digest = replay._atomic_write_document(path, document)

    assert digest == replay._sha256_bytes(replay._document_bytes(document))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(path.parent.glob(".*.tmp")) == []


def test_artifact_set_is_published_once_as_a_complete_private_generation(
    replay, tmp_path: Path
):
    output_directory = tmp_path / "generation"
    documents = {
        "corpus": {"kind": "corpus", "samples": []},
        "baseline": {"kind": "baseline", "samples": []},
        "candidate": {"kind": "candidate", "samples": []},
    }

    hashes = replay._atomic_write_artifact_set(output_directory, documents)

    assert set(hashes) == {"corpus", "baseline", "candidate"}
    assert output_directory.is_dir()
    assert {path.name for path in output_directory.iterdir()} == set(
        replay.ARTIFACT_FILENAMES.values()
    )
    original_payloads = {}
    for name, filename in replay.ARTIFACT_FILENAMES.items():
        path = output_directory / filename
        original_payloads[name] = path.read_bytes()
        assert hashes[name] == replay._sha256_bytes(original_payloads[name])
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".phone-crop-replay.*.tmp")) == []

    with pytest.raises(replay.ReplayInputError, match="must not already exist"):
        replay._atomic_write_artifact_set(output_directory, documents)

    assert {
        name: (output_directory / filename).read_bytes()
        for name, filename in replay.ARTIFACT_FILENAMES.items()
    } == original_payloads


def test_source_snapshot_rejects_dirty_or_unstable_provenance(replay, monkeypatch):
    clean = _source_provenance()
    script_hashes = iter(["a" * 64, "a" * 64])
    monkeypatch.setattr(replay, "_script_sha256", lambda: next(script_hashes))
    monkeypatch.setattr(replay, "_git_source_provenance", lambda: clean)
    assert replay._capture_source_snapshot(require_clean=True) == {
        "scriptSha256": "a" * 64,
        "sourceTree": clean,
    }

    dirty = {**clean, "gitDirty": True}
    monkeypatch.setattr(replay, "_script_sha256", lambda: "b" * 64)
    monkeypatch.setattr(replay, "_git_source_provenance", lambda: dirty)
    with pytest.raises(replay.ReplayInputError, match="clean Git worktree"):
        replay._capture_source_snapshot(require_clean=True)

    changing_hashes = iter(["c" * 64, "d" * 64])
    monkeypatch.setattr(replay, "_script_sha256", lambda: next(changing_hashes))
    monkeypatch.setattr(replay, "_git_source_provenance", lambda: clean)
    with pytest.raises(replay.ReplayInputError, match="script changed"):
        replay._capture_source_snapshot(require_clean=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("cuda", "cuda"), ("CUDA:0", "cuda:0"), ("0", "0"), ("cpu", "cpu")],
)
def test_device_values_are_bounded_and_safe(replay, value: str, expected: str):
    assert replay._validated_device(value) == expected


def test_device_rejects_serializable_urls_or_credentials(replay):
    with pytest.raises(replay.ReplayInputError, match="device must be"):
        replay._validated_device("cuda;https://user:secret@example.test")
