"""Tests for route obstruction snapshot telemetry."""

import obstruction_analytics
import state


def test_obstruction_snapshot_counts_objects_inside_route_zone():
    camera_id = "route_cam"
    obstruction_analytics.reset_obstruction_state(camera_id)
    camera = {
        "id": camera_id,
        "name": "Route Camera",
        "zone": "Gate Lane",
        "capabilities": ["route_obstruction"],
        "obstruction_threshold": 2,
        "zones": [
            {
                "id": "route_zone",
                "name": "Keep Clear Route",
                "type": "route",
                "analytics": "obstruction",
                "classes": ["car", "person"],
                "points": [[0.0, 0.0], [0.7, 0.0], [0.7, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "car", "confidence": 0.91, "bbox": [100, 100, 300, 260]},
        {"class": "person", "confidence": 0.88, "bbox": [500, 120, 620, 420]},
        {"class": "truck", "confidence": 0.87, "bbox": [900, 120, 1040, 420]},
        {"class": "chair", "confidence": 0.82, "bbox": [120, 500, 220, 620]},
    ]

    snapshot = obstruction_analytics.get_obstruction_snapshot(camera_id, camera)

    assert snapshot["objectCount"] == 2
    assert snapshot["maxZoneCount"] == 2
    assert snapshot["obstructionActive"] is True
    assert snapshot["calibrated"] is False
    assert snapshot["routeZones"][0]["objectCount"] == 2
    assert snapshot["routeZones"][0]["classCounts"] == {"car": 1, "person": 1}


def test_obstruction_snapshot_reports_calibrated_severity():
    camera_id = "route_cam_severity"
    obstruction_analytics.reset_obstruction_state(camera_id)
    camera = {
        "id": camera_id,
        "name": "Route Camera",
        "zone": "Gate Lane",
        "capabilities": ["route_obstruction"],
        "obstruction_threshold": 2,
        "zones": [
            {
                "id": "route_zone",
                "name": "Keep Clear Route",
                "type": "route",
                "analytics": "obstruction",
                "classes": ["car", "person"],
                "threshold": 2,
                "area_square_meters": 10.0,
                "severity_thresholds": {
                    "medium_density_objects_per_square_meter": 0.2,
                    "high_density_objects_per_square_meter": 0.3,
                },
                "points": [[0.0, 0.0], [0.8, 0.0], [0.8, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "car", "confidence": 0.91, "bbox": [100, 100, 300, 260]},
        {"class": "person", "confidence": 0.88, "bbox": [500, 120, 620, 420]},
        {"class": "truck", "confidence": 0.87, "bbox": [900, 120, 1040, 420]},
    ]

    snapshot = obstruction_analytics.get_obstruction_snapshot(camera_id, camera)
    route_zone = snapshot["routeZones"][0]

    assert snapshot["calibrated"] is True
    assert snapshot["calibratedZoneCount"] == 1
    assert snapshot["maxObstructionDensityObjectsPerSquareMeter"] == 0.2
    assert snapshot["maxSeverity"] == "medium"
    assert snapshot["maxSeverityRank"] == 2
    assert route_zone["areaSquareMeters"] == 10.0
    assert route_zone["obstructionDensityObjectsPerSquareMeter"] == 0.2
    assert route_zone["severity"] == "medium"
    assert route_zone["severityThresholds"]["mediumDensityObjectsPerSquareMeter"] == 0.2


def test_obstruction_snapshot_without_zones_counts_configured_classes():
    camera_id = "route_cam_no_zones"
    obstruction_analytics.reset_obstruction_state(camera_id)
    camera = {
        "id": camera_id,
        "name": "Route Camera",
        "zone": "Gate Lane",
        "capabilities": ["route_obstruction"],
        "obstruction_classes": ["car", "truck"],
        "obstruction_threshold": 3,
        "zones": [],
    }
    state.camera_detections[camera_id] = [
        {"class": "car", "confidence": 0.91, "bbox": [100, 100, 300, 260]},
        {"class": "truck", "confidence": 0.88, "bbox": [500, 120, 620, 420]},
        {"class": "person", "confidence": 0.82, "bbox": [120, 500, 220, 620]},
    ]

    snapshot = obstruction_analytics.get_obstruction_snapshot(camera_id, camera)

    assert snapshot["objectCount"] == 2
    assert snapshot["maxZoneCount"] == 2
    assert snapshot["obstructionActive"] is False


def test_obstruction_snapshot_tracks_active_duration(monkeypatch):
    camera_id = "route_duration_cam"
    obstruction_analytics.reset_obstruction_state(camera_id)
    now = 1000.0
    monkeypatch.setattr(obstruction_analytics.time, "time", lambda: now)
    camera = {
        "id": camera_id,
        "name": "Route Camera",
        "zone": "Gate Lane",
        "capabilities": ["route_obstruction"],
        "obstruction_threshold": 2,
        "obstruction_min_duration_seconds": 5,
        "zones": [
            {
                "id": "route_zone",
                "name": "Keep Clear Route",
                "type": "route",
                "analytics": "obstruction",
                "classes": ["car", "person"],
                "points": [[0.0, 0.0], [0.9, 0.0], [0.9, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "car", "confidence": 0.91, "bbox": [100, 100, 300, 260]},
        {"class": "person", "confidence": 0.88, "bbox": [500, 120, 620, 420]},
    ]

    first = obstruction_analytics.get_obstruction_snapshot(camera_id, camera)
    now = 1006.0
    second = obstruction_analytics.get_obstruction_snapshot(camera_id, camera)

    assert first["obstructionActive"] is True
    assert second["activeSeconds"] == 6
    assert second["durationReady"] is True
    assert second["routeZones"][0]["activeSeconds"] == 6
    assert second["routeZones"][0]["durationReady"] is True


def test_obstruction_duration_session_resets_after_long_gap(monkeypatch):
    camera_id = "route_duration_reset_cam"
    obstruction_analytics.reset_obstruction_state(camera_id)
    now = 2000.0
    monkeypatch.setattr(obstruction_analytics.time, "time", lambda: now)
    camera = {
        "id": camera_id,
        "name": "Route Camera",
        "zone": "Gate Lane",
        "capabilities": ["route_obstruction"],
        "obstruction_threshold": 1,
        "obstruction_session_reset_gap_seconds": 10,
        "zones": [],
    }
    state.camera_detections[camera_id] = [
        {"class": "car", "confidence": 0.91, "bbox": [100, 100, 300, 260]},
    ]

    obstruction_analytics.get_obstruction_snapshot(camera_id, camera)
    now = 2012.0
    snapshot = obstruction_analytics.get_obstruction_snapshot(camera_id, camera)

    assert snapshot["obstructionActive"] is True
    assert snapshot["activeSeconds"] == 0
    assert snapshot["sessionSeconds"] == 0
