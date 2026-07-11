from copy import deepcopy

import cv2
import numpy as np
import pytest

import state
import video_processing
from mjpeg_fanout import MjpegFanout


@pytest.fixture(autouse=True)
def camera_config(monkeypatch):
    monkeypatch.setattr(video_processing, "stream_fanout", MjpegFanout())
    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: {"cameras": {"cam1": {"name": "Camera 1", "zones": []}}},
    )
    yield
    state.camera_frames.pop("cam1", None)
    state.camera_clean_frames.pop("cam1", None)


def test_render_stream_views_resizes_source_once_and_preserves_clean_pixels(monkeypatch):
    frame = np.arange(900 * 1600 * 3, dtype=np.uint8).reshape((900, 1600, 3))
    detections = [{"class": "person", "confidence": 0.9, "bbox": [160, 90, 1440, 810]}]
    original_detections = deepcopy(detections)
    original_resize = video_processing.cv2.resize
    resize_calls = []

    def tracked_resize(*args, **kwargs):
        resize_calls.append((args, kwargs))
        return original_resize(*args, **kwargs)

    expected_clean = original_resize(frame, (854, 480))
    monkeypatch.setattr(video_processing.cv2, "resize", tracked_resize)

    annotated, clean = video_processing._render_stream_views("cam1", frame, detections)

    assert len(resize_calls) == 1
    assert annotated.shape == clean.shape == (480, 854, 3)
    assert np.array_equal(clean, expected_clean)
    assert detections == original_detections


def test_render_stream_views_preserves_coco_style_metadata_and_count(monkeypatch):
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    detections = [
        {
            "class_id": 2,
            "class": "car",
            "confidence": 0.9,
            "bbox": [160, 90, 1440, 810],
            "model_family": "coco_primary",
        },
        {
            "class_id": 0,
            "class": "helmet",
            "confidence": 0.8,
            "bbox": [200, 100, 400, 300],
            "model_family": "ppe_specialist",
        },
    ]
    captured = {}

    def capture_records(render_frame, records, camera_id, **kwargs):
        captured["records"] = records
        captured["camera_id"] = camera_id
        captured["show_overlay"] = kwargs["show_overlay"]
        return render_frame.copy(), []

    def capture_overlay(render_frame, **kwargs):
        captured["count"] = kwargs["detection_count"]
        return render_frame

    monkeypatch.setattr(video_processing, "draw_detection_records", capture_records)
    monkeypatch.setattr(video_processing, "apply_camera_overlay", capture_overlay)

    video_processing._render_stream_views("cam1", frame, detections)

    assert captured == {
        "records": [{
            "class_id": 2,
            "class": "car",
            "confidence": 0.9,
            "bbox": [85, 48, 769, 432],
            "model_family": "coco_primary",
        }],
        "camera_id": "cam1",
        "show_overlay": False,
        "count": 1,
    }


def test_scale_stream_detection_records_scales_clamps_and_does_not_mutate():
    detections = [
        {"class": "person", "confidence": 0.9, "bbox": [160, 90, 1440, 810]},
        {"class": "truck", "confidence": 0.8, "bbox": [-20, -10, 1700, 950]},
        {"class": "invalid", "confidence": 0.5},
    ]
    original = deepcopy(detections)

    scaled = video_processing._scale_stream_detection_records(
        detections,
        (900, 1600, 3),
        (480, 854, 3),
    )

    assert [record["bbox"] for record in scaled] == [
        [85, 48, 769, 432],
        [0, 0, 853, 479],
    ]
    assert detections == original


def test_render_stream_views_skips_resize_and_coordinate_scaling_at_native_width(monkeypatch):
    frame = np.zeros((288, 352, 3), dtype=np.uint8)
    detections = [{"class": "person", "confidence": 0.9, "bbox": [35, 29, 317, 259]}]
    original = deepcopy(detections)
    monkeypatch.setattr(
        video_processing.cv2,
        "resize",
        lambda *_args, **_kwargs: pytest.fail("native stream frame should not be resized"),
    )
    monkeypatch.setattr(
        video_processing,
        "_scale_stream_detection_records",
        lambda *_args, **_kwargs: pytest.fail("native coordinates should not be scaled"),
    )

    annotated, clean = video_processing._render_stream_views("cam1", frame, detections)

    assert annotated.shape == clean.shape == frame.shape
    assert clean is frame
    assert detections == original


def test_publish_stream_frame_encodes_matching_dimensions_and_keeps_source_coordinates():
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    detections = [{"class": "person", "confidence": 0.9, "bbox": [160, 90, 1440, 810]}]
    original = deepcopy(detections)

    try:
        video_processing._publish_stream_frame(
            "cam1",
            frame,
            detections,
            jpeg_quality=70,
        )

        annotated = cv2.imdecode(np.frombuffer(state.camera_frames["cam1"], np.uint8), cv2.IMREAD_COLOR)
        clean = cv2.imdecode(np.frombuffer(state.camera_clean_frames["cam1"], np.uint8), cv2.IMREAD_COLOR)
        assert annotated.shape == clean.shape == (480, 854, 3)
        assert not np.array_equal(annotated, clean)
        assert detections == original
    finally:
        state.camera_frames.pop("cam1", None)
        state.camera_clean_frames.pop("cam1", None)


def test_publish_stream_frame_preserves_non_bbox_source_annotations(monkeypatch):
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    source_annotated = np.zeros((720, 1280, 3), dtype=np.uint8)
    source_annotated[100:200, 100:200] = (0, 0, 255)
    monkeypatch.setattr(
        video_processing,
        "_render_stream_views",
        lambda *_args, **_kwargs: pytest.fail("specialist annotations should use the preserved source view"),
    )

    try:
        video_processing._publish_stream_frame(
            "cam1",
            frame,
            [],
            jpeg_quality=70,
            source_annotated=source_annotated,
        )

        annotated = cv2.imdecode(np.frombuffer(state.camera_frames["cam1"], np.uint8), cv2.IMREAD_COLOR)
        clean = cv2.imdecode(np.frombuffer(state.camera_clean_frames["cam1"], np.uint8), cv2.IMREAD_COLOR)
        assert annotated.shape == clean.shape == (480, 854, 3)
        assert not np.array_equal(annotated, clean)
    finally:
        state.camera_frames.pop("cam1", None)
        state.camera_clean_frames.pop("cam1", None)


def test_preserved_source_annotation_uses_native_and_specialist_views_only():
    native_frame = np.zeros((288, 352, 3), dtype=np.uint8)
    native_annotation = native_frame.copy()
    large_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    large_annotation = large_frame.copy()

    assert video_processing._preserved_source_annotation(native_frame, native_annotation, {}) is native_annotation
    assert video_processing._preserved_source_annotation(large_frame, large_annotation, {}) is None
    assert video_processing._preserved_source_annotation(
        large_frame,
        large_annotation,
        {"run_pose_specialist": True},
    ) is large_annotation
    assert video_processing._preserved_source_annotation(
        large_frame,
        np.zeros((720, 1280, 3), dtype=np.uint8),
        {"run_face_recognition": True},
    ) is None


def test_encode_stream_jpeg_rejects_encoder_failure(monkeypatch):
    monkeypatch.setattr(video_processing.cv2, "imencode", lambda *_args, **_kwargs: (False, None))

    with pytest.raises(RuntimeError, match="Failed to encode stream frame"):
        video_processing._encode_stream_jpeg(np.zeros((10, 10, 3), dtype=np.uint8), 70)


def test_publish_stream_frame_does_not_publish_a_partial_jpeg_pair(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    encode_calls = 0

    def fail_second_encode(_frame, _jpeg_quality):
        nonlocal encode_calls
        encode_calls += 1
        if encode_calls == 2:
            raise RuntimeError("clean encode failed")
        return b"new-annotated"

    monkeypatch.setattr(video_processing, "_render_stream_views", lambda *_args: (frame, frame))
    monkeypatch.setattr(video_processing, "_encode_stream_jpeg", fail_second_encode)
    monkeypatch.setattr(
        video_processing.stream_fanout,
        "publish",
        lambda *_args, **_kwargs: pytest.fail("partial frame pair must not reach fanout"),
    )
    monkeypatch.setitem(state.camera_frames, "cam1", b"old-annotated")
    monkeypatch.setitem(state.camera_clean_frames, "cam1", b"old-clean")

    with pytest.raises(RuntimeError, match="clean encode failed"):
        video_processing._publish_stream_frame("cam1", frame, [], jpeg_quality=70)

    assert state.camera_frames["cam1"] == b"old-annotated"
    assert state.camera_clean_frames["cam1"] == b"old-clean"


def test_publish_stream_frame_notifies_fanout_after_both_caches_update(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    encoded = iter((b"new-annotated", b"new-clean"))
    publications = []

    def capture_publication(camera_id, jpeg):
        publications.append(
            (
                camera_id,
                jpeg,
                state.camera_frames.get(camera_id),
                state.camera_clean_frames.get(camera_id),
            )
        )

    monkeypatch.setattr(video_processing, "_render_stream_views", lambda *_args: (frame, frame))
    monkeypatch.setattr(video_processing, "_encode_stream_jpeg", lambda *_args: next(encoded))
    monkeypatch.setattr(video_processing.stream_fanout, "publish", capture_publication)

    video_processing._publish_stream_frame("cam1", frame, [], jpeg_quality=70)

    assert publications == [
        ("cam1", b"new-annotated", b"new-annotated", b"new-clean")
    ]


def test_publish_stream_frame_reuses_fresh_clean_cache(monkeypatch):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    encode_calls = []

    monkeypatch.setattr(
        video_processing,
        "_render_stream_views",
        lambda *_args: (frame, frame),
    )

    def encode_annotated(_frame, _jpeg_quality):
        encode_calls.append(True)
        return b"new-annotated"

    monkeypatch.setattr(video_processing, "_encode_stream_jpeg", encode_annotated)

    clean_jpeg, clean_encoded = video_processing._publish_stream_frame(
        "cam1",
        frame,
        [],
        jpeg_quality=70,
        cached_clean_jpeg=b"cached-clean",
    )

    assert encode_calls == [True]
    assert clean_jpeg == b"cached-clean"
    assert clean_encoded is False
    assert state.camera_frames["cam1"] == b"new-annotated"
    assert state.camera_clean_frames["cam1"] == b"cached-clean"


def test_stream_clean_cache_refreshes_at_one_second():
    assert video_processing._stream_clean_cache_due(None, 10.0, 10.25) is True
    assert video_processing._stream_clean_cache_due(b"clean", 0.0, 10.25) is True
    assert video_processing._stream_clean_cache_due(b"clean", 10.0, 10.99) is False
    assert video_processing._stream_clean_cache_due(b"clean", 10.0, 11.0) is True


def test_idle_empty_publication_encodes_clean_frame_once(monkeypatch):
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    encode_calls = []
    publications = []
    monkeypatch.setattr(
        video_processing,
        "_render_stream_views",
        lambda *_args, **_kwargs: pytest.fail("idle empty frame should not be annotated"),
    )

    def encode_once(clean_view, jpeg_quality):
        encode_calls.append((clean_view.shape, jpeg_quality))
        return b"shared-clean"

    monkeypatch.setattr(video_processing, "_encode_stream_jpeg", encode_once)
    monkeypatch.setattr(
        video_processing.stream_fanout,
        "publish",
        lambda camera_id, jpeg: publications.append((camera_id, jpeg)),
    )

    clean_jpeg, clean_encoded = video_processing._publish_stream_frame(
        "cam1",
        frame,
        [],
        jpeg_quality=70,
        annotation_required=False,
        cached_clean_jpeg=b"stale-clean",
    )

    assert encode_calls == [((480, 854, 3), 70)]
    assert clean_jpeg == b"shared-clean"
    assert clean_encoded is True
    assert state.camera_frames["cam1"] == b"shared-clean"
    assert state.camera_clean_frames["cam1"] == b"shared-clean"
    assert publications == [("cam1", b"shared-clean")]


def test_inference_snapshot_renders_large_bbox_frame_when_source_annotation_was_skipped():
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    detections = [
        {
            "class_id": 0,
            "class": "person",
            "confidence": 0.9,
            "bbox": [160, 90, 1440, 810],
            "model_family": "coco_primary",
        },
    ]
    original = deepcopy(detections)

    annotated_jpeg, clean_jpeg = video_processing._encode_inference_snapshot_pair(
        "cam1",
        frame,
        detections,
        70,
        annotated_frame=None,
    )

    annotated = cv2.imdecode(np.frombuffer(annotated_jpeg, np.uint8), cv2.IMREAD_COLOR)
    clean = cv2.imdecode(np.frombuffer(clean_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert annotated.shape == clean.shape == (480, 854, 3)
    assert not np.array_equal(annotated, clean)
    assert detections == original
