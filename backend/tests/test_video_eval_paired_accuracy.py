"""Tests for paired labeled accuracy comparison in the video evaluation runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
VIDEO_EVAL_PATH = ROOT / "scripts" / "video_eval.py"


def _load_video_eval():
    spec = importlib.util.spec_from_file_location(
        "video_eval_paired_test", VIDEO_EVAL_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _documents():
    corpus = {
        "corpus_id": "paired-fixture-v1",
        "rules": ["ppe_missing", "phone_use"],
        "samples": [
            {
                "sample_id": "c1-0000",
                "clip_id": "clip-1",
                "timestamp_ms": 0,
                "labels": {"ppe_missing": True, "phone_use": False},
            },
            {
                "sample_id": "c1-1000",
                "clip_id": "clip-1",
                "timestamp_ms": 1000,
                "labels": {"ppe_missing": True, "phone_use": False},
            },
            {
                "sample_id": "c2-0000",
                "clip_id": "clip-2",
                "timestamp_ms": 0,
                "labels": {"ppe_missing": False, "phone_use": True},
            },
            {
                "sample_id": "c2-1000",
                "clip_id": "clip-2",
                "timestamp_ms": 1000,
                "labels": {"ppe_missing": False, "phone_use": False},
            },
        ],
        "events": [
            {
                "event_id": "ppe-event",
                "clip_id": "clip-1",
                "rule": "ppe_missing",
                "start_ms": 0,
                "end_ms": 1000,
            },
            {
                "event_id": "phone-event",
                "clip_id": "clip-2",
                "rule": "phone_use",
                "start_ms": 0,
                "end_ms": 500,
            },
        ],
    }

    def sample_predictions(values):
        return [
            {
                "sample_id": sample["sample_id"],
                "clip_id": sample["clip_id"],
                "timestamp_ms": sample["timestamp_ms"],
                "predictions": prediction,
            }
            for sample, prediction in zip(corpus["samples"], values)
        ]

    baseline = {
        "run_id": "baseline",
        "corpus_id": corpus["corpus_id"],
        "samples": sample_predictions(
            [
                {"ppe_missing": True, "phone_use": False},
                {"ppe_missing": False, "phone_use": False},
                {"ppe_missing": True, "phone_use": False},
                {"ppe_missing": False, "phone_use": False},
            ]
        ),
        "alerts": [
            {
                "alert_id": "base-match",
                "clip_id": "clip-1",
                "rule": "ppe_missing",
                "timestamp_ms": 100,
            },
            {
                "alert_id": "base-duplicate",
                "clip_id": "clip-1",
                "rule": "ppe_missing",
                "timestamp_ms": 200,
            },
            {
                "alert_id": "base-false",
                "clip_id": "clip-2",
                "rule": "ppe_missing",
                "timestamp_ms": 900,
            },
        ],
    }
    candidate = {
        "run_id": "candidate",
        "corpus_id": corpus["corpus_id"],
        "samples": sample_predictions(
            [
                {"ppe_missing": True, "phone_use": False},
                {"ppe_missing": True, "phone_use": False},
                {"ppe_missing": False, "phone_use": True},
                {"ppe_missing": False, "phone_use": False},
            ]
        ),
        "alerts": [
            {
                "alert_id": "candidate-ppe",
                "clip_id": "clip-1",
                "rule": "ppe_missing",
                "timestamp_ms": 100,
            },
            {
                "alert_id": "candidate-phone",
                "clip_id": "clip-2",
                "rule": "phone_use",
                "timestamp_ms": 100,
            },
        ],
    }
    return corpus, baseline, candidate


def _normalized_report(video_eval):
    corpus_document, baseline_document, candidate_document = _documents()
    corpus = video_eval.validate_paired_accuracy_corpus(corpus_document)
    baseline = video_eval.validate_paired_accuracy_run(
        baseline_document, corpus, "baseline"
    )
    candidate = video_eval.validate_paired_accuracy_run(
        candidate_document, corpus, "candidate"
    )
    return video_eval.build_paired_accuracy_report(
        corpus,
        baseline,
        candidate,
        corpus_sha256="c" * 64,
        baseline_sha256="b" * 64,
        candidate_sha256="a" * 64,
        bootstrap_resamples=200,
        bootstrap_seed=17,
        generated_at="2026-07-15T00:00:00+00:00",
    )


def test_paired_report_computes_frame_event_and_duplicate_metrics():
    video_eval = _load_video_eval()
    report = _normalized_report(video_eval)

    baseline_ppe = report["frame_rule_metrics"]["ppe_missing"]["baseline"]
    candidate_ppe = report["frame_rule_metrics"]["ppe_missing"]["candidate"]
    assert baseline_ppe == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "sample_count": 4,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert candidate_ppe["precision"] == 1.0
    assert candidate_ppe["recall"] == 1.0
    assert candidate_ppe["f1"] == 1.0
    assert (
        report["frame_rule_metrics"]["ppe_missing"]["paired_change"]["f1"]["estimate"]
        == 0.5
    )

    overall = report["alert_event_metrics"]["overall"]
    assert overall["baseline"]["event_recall"] == 0.5
    assert overall["baseline"]["duplicate_alert_count"] == 1
    assert overall["baseline"]["duplicate_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert overall["baseline"]["false_alert_count"] == 1
    assert overall["candidate"]["event_recall"] == 1.0
    assert overall["candidate"]["duplicate_rate"] == 0.0
    assert overall["paired_change"]["event_recall"]["estimate"] == 0.5
    assert report["methodology"]["aggregate_count_or_fps_inference"] is False


def test_paired_bootstrap_intervals_are_deterministic():
    video_eval = _load_video_eval()

    first = _normalized_report(video_eval)
    second = _normalized_report(video_eval)

    assert first == second
    interval = first["frame_rule_metrics"]["ppe_missing"]["paired_change"]["f1"]
    assert interval["interval_available"] is True
    assert interval["valid_resamples"] <= interval["total_resamples"] == 200
    assert interval["ci_lower"] <= interval["estimate"] <= interval["ci_upper"]


def test_event_matching_maximizes_overlapping_event_recall_before_counting_duplicates():
    video_eval = _load_video_eval()
    events = [
        {"event_id": "early", "start_ms": 0, "end_ms": 10},
        {"event_id": "late", "start_ms": 5, "end_ms": 20},
    ]
    alerts = [
        {"alert_id": "a", "timestamp_ms": 6},
        {"alert_id": "b", "timestamp_ms": 15},
        {"alert_id": "c", "timestamp_ms": 18},
    ]

    counts = video_eval._paired_match_event_alerts(
        events,
        alerts,
        alert_early_ms=0,
        alert_late_ms=0,
    )

    assert counts["recalled_event_count"] == 2
    assert counts["duplicate_alert_count"] == 1
    assert counts["false_alert_count"] == 0


def test_paired_run_rejects_missing_or_misaligned_timestamped_samples():
    video_eval = _load_video_eval()
    corpus_document, baseline_document, _candidate_document = _documents()
    corpus = video_eval.validate_paired_accuracy_corpus(corpus_document)
    baseline_document["samples"] = baseline_document["samples"][:-1]

    with pytest.raises(ValueError, match="missing corpus samples"):
        video_eval.validate_paired_accuracy_run(baseline_document, corpus, "baseline")

    _corpus_document, baseline_document, _candidate_document = _documents()
    baseline_document["samples"][0]["timestamp_ms"] = 1
    with pytest.raises(ValueError, match="does not match the corpus clip/timestamp"):
        video_eval.validate_paired_accuracy_run(baseline_document, corpus, "baseline")

    _corpus_document, baseline_document, _candidate_document = _documents()
    baseline_document["samples"][0]["predictions"]["ppe_missing"] = 1
    with pytest.raises(ValueError, match="must be an explicit boolean"):
        video_eval.validate_paired_accuracy_run(baseline_document, corpus, "baseline")


def test_paired_accuracy_cli_writes_private_atomic_aggregate_json(tmp_path):
    video_eval = _load_video_eval()
    corpus, baseline, candidate = _documents()
    corpus_path = tmp_path / "corpus.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "results" / "paired.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    exit_code = video_eval.main(
        [
            "paired-accuracy",
            "--corpus",
            str(corpus_path),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--out",
            str(output_path),
            "--bootstrap-resamples",
            "100",
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "rakshak.paired_accuracy.v1"
    assert report["corpus"]["sample_count"] == 4
    assert len(report["corpus"]["sha256"]) == 64
    assert output_path.stat().st_mode & 0o777 == 0o600
    rendered = output_path.read_text(encoding="utf-8")
    assert '"sample_id"' not in rendered
    assert '"predictions"' not in rendered
    assert not list(output_path.parent.glob(".*.tmp"))
