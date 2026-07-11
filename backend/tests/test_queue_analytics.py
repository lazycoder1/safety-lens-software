"""Tests for queue snapshot telemetry."""

import pytest

import object_lifecycle_analytics
import obstruction_analytics
import occupancy_analytics
import queue_analytics
import state


def test_queue_snapshot_counts_people_inside_queue_zone():
    camera_id = "queue_cam"
    queue_analytics.reset_queue_state(camera_id)
    camera = {
        "id": camera_id,
        "name": "Queue Camera",
        "zone": "Lobby",
        "capabilities": ["queue_monitoring"],
        "queue_threshold": 2,
        "zones": [
            {
                "id": "queue_zone",
                "name": "Lobby Queue",
                "type": "queue",
                "points": [[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "person", "confidence": 0.91, "bbox": [100, 100, 200, 400]},
        {"class": "person", "confidence": 0.88, "bbox": [300, 120, 420, 420]},
        {"class": "person", "confidence": 0.87, "bbox": [900, 120, 1040, 420]},
        {"class": "chair", "confidence": 0.82, "bbox": [120, 500, 220, 620]},
    ]

    snapshot = queue_analytics.get_queue_snapshot(camera_id, camera)

    assert snapshot["personCount"] == 2
    assert snapshot["maxZoneCount"] == 2
    assert snapshot["queueActive"] is True
    assert snapshot["calibrated"] is False
    assert snapshot["queueZones"][0]["personCount"] == 2


def test_queue_snapshot_without_zones_counts_all_people():
    camera_id = "queue_cam_no_zones"
    queue_analytics.reset_queue_state(camera_id)
    camera = {
        "id": camera_id,
        "name": "Queue Camera",
        "zone": "Lobby",
        "capabilities": ["queue_monitoring"],
        "queue_threshold": 3,
        "zones": [],
    }
    state.camera_detections[camera_id] = [
        {"class": "person", "confidence": 0.91, "bbox": [100, 100, 200, 400]},
        {"class": "person", "confidence": 0.88, "bbox": [300, 120, 420, 420]},
    ]

    snapshot = queue_analytics.get_queue_snapshot(camera_id, camera)

    assert snapshot["personCount"] == 2
    assert snapshot["maxZoneCount"] == 2
    assert snapshot["queueActive"] is False


def test_queue_snapshot_tracks_active_duration(monkeypatch):
    camera_id = "queue_duration_cam"
    queue_analytics.reset_queue_state(camera_id)
    now = 1000.0
    monkeypatch.setattr(queue_analytics.time, "time", lambda: now)
    camera = {
        "id": camera_id,
        "name": "Queue Camera",
        "zone": "Lobby",
        "capabilities": ["queue_monitoring"],
        "queue_threshold": 2,
        "queue_min_duration_seconds": 5,
        "zones": [
            {
                "id": "queue_zone",
                "name": "Lobby Queue",
                "type": "queue",
                "points": [[0.0, 0.0], [0.9, 0.0], [0.9, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "person", "confidence": 0.91, "bbox": [100, 100, 200, 400]},
        {"class": "person", "confidence": 0.88, "bbox": [300, 120, 420, 420]},
    ]

    first = queue_analytics.get_queue_snapshot(camera_id, camera)
    now = 1006.0
    second = queue_analytics.get_queue_snapshot(camera_id, camera)

    assert first["queueActive"] is True
    assert second["activeSeconds"] == 6
    assert second["durationReady"] is True
    assert second["queueZones"][0]["activeSeconds"] == 6
    assert second["queueZones"][0]["durationReady"] is True


def test_queue_snapshot_reports_calibrated_people_density():
    camera_id = "queue_density_cam"
    queue_analytics.reset_queue_state(camera_id)
    camera = {
        "id": camera_id,
        "name": "Queue Density Camera",
        "zone": "Lobby",
        "capabilities": ["queue_monitoring"],
        "queue_threshold": 99,
        "zones": [
            {
                "id": "queue_zone",
                "name": "Lobby Queue",
                "type": "queue",
                "area_square_meters": 4.0,
                "density_threshold_people_per_square_meter": 0.5,
                "points": [[0.0, 0.0], [0.8, 0.0], [0.8, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "person", "confidence": 0.91, "bbox": [100, 100, 200, 400]},
        {"class": "person", "confidence": 0.88, "bbox": [300, 120, 420, 420]},
        {"class": "chair", "confidence": 0.82, "bbox": [120, 500, 220, 620]},
    ]

    snapshot = queue_analytics.get_queue_snapshot(camera_id, camera)

    assert snapshot["calibrated"] is True
    assert snapshot["calibratedZoneCount"] == 1
    assert snapshot["queueActive"] is True
    assert snapshot["maxDensityPeoplePerSquareMeter"] == 0.5
    assert snapshot["densityThresholdPeoplePerSquareMeter"] == 0.5
    assert snapshot["queueZones"][0]["areaSquareMeters"] == 4.0
    assert snapshot["queueZones"][0]["densityPeoplePerSquareMeter"] == 0.5


def test_queue_snapshot_reports_configured_wait_time_tracking(monkeypatch):
    camera_id = "queue_wait_cam"
    queue_analytics.reset_queue_state(camera_id)
    now = 3000.0
    monkeypatch.setattr(queue_analytics.time, "time", lambda: now)
    camera = {
        "id": camera_id,
        "name": "Queue Wait Camera",
        "zone": "Lobby",
        "capabilities": ["queue_monitoring"],
        "queue_threshold": 2,
        "queue_wait_track_max_distance": 0.3,
        "zones": [
            {
                "id": "queue_zone",
                "name": "Lobby Queue",
                "type": "queue",
                "wait_tracking_enabled": True,
                "wait_threshold_seconds": 5,
                "points": [[0.0, 0.0], [0.9, 0.0], [0.9, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "person", "confidence": 0.91, "bbox": [100, 100, 200, 400]},
        {"class": "person", "confidence": 0.88, "bbox": [300, 120, 420, 420]},
    ]

    first = queue_analytics.get_queue_snapshot(camera_id, camera)
    now = 3006.0
    second = queue_analytics.get_queue_snapshot(camera_id, camera)

    assert first["waitTimeTrackingEnabled"] is True
    assert first["trackedPersonCount"] == 2
    assert first["maxWaitSeconds"] == 0
    assert second["trackedPersonCount"] == 2
    assert second["maxWaitSeconds"] == 6
    assert second["averageWaitSeconds"] == 6.0
    assert second["waitTimeReady"] is True
    assert second["queueZones"][0]["maxWaitSeconds"] == 6
    assert second["queueZones"][0]["waitTimeReady"] is True


def test_queue_duration_session_resets_after_long_gap(monkeypatch):
    camera_id = "queue_duration_reset_cam"
    queue_analytics.reset_queue_state(camera_id)
    now = 2000.0
    monkeypatch.setattr(queue_analytics.time, "time", lambda: now)
    camera = {
        "id": camera_id,
        "name": "Queue Camera",
        "zone": "Lobby",
        "capabilities": ["queue_monitoring"],
        "queue_threshold": 1,
        "queue_session_reset_gap_seconds": 10,
        "zones": [],
    }
    state.camera_detections[camera_id] = [
        {"class": "person", "confidence": 0.91, "bbox": [100, 100, 200, 400]},
    ]

    queue_analytics.get_queue_snapshot(camera_id, camera)
    now = 2012.0
    snapshot = queue_analytics.get_queue_snapshot(camera_id, camera)

    assert snapshot["queueActive"] is True
    assert snapshot["activeSeconds"] == 0
    assert snapshot["sessionSeconds"] == 0


@pytest.mark.parametrize(
    "analytics_module",
    [
        queue_analytics,
        obstruction_analytics,
        occupancy_analytics,
        object_lifecycle_analytics,
    ],
)
def test_camera_frame_size_prefers_cached_detection_dimensions(monkeypatch, analytics_module):
    monkeypatch.setattr(state, "camera_frame_dimensions", {"cam1": (960, 540)})
    monkeypatch.setattr(state, "camera_clean_frames", {"cam1": b"not-a-jpeg"})
    monkeypatch.setattr(
        analytics_module.cv2,
        "imdecode",
        lambda *_args, **_kwargs: pytest.fail("cached dimensions should avoid JPEG decode"),
    )

    assert analytics_module._camera_frame_size({"id": "cam1"}) == (960, 540)
