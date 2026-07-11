"""Tests for workstation occupancy duration telemetry."""

import occupancy_analytics
import state


def test_occupancy_report_tracks_zone_occupied_duration(monkeypatch):
    camera_id = "occupancy_duration_cam"
    occupancy_analytics._sessions.pop(camera_id, None)
    now = 1000.0
    monkeypatch.setattr(occupancy_analytics.time, "time", lambda: now)
    camera = {
        "id": camera_id,
        "name": "Office Occupancy",
        "zone": "Office",
        "capabilities": ["office_occupancy"],
        "occupancy_min_duration_seconds": 5,
        "zones": [
            {
                "id": "desk_zone",
                "name": "Desk Zone",
                "type": "workstation",
                "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            }
        ],
    }
    state.camera_detections[camera_id] = [
        {"class": "person", "confidence": 0.91, "bbox": [100, 100, 240, 420]},
    ]

    first = occupancy_analytics.get_occupancy_report(camera_id, camera)
    now = 1006.0
    second = occupancy_analytics.get_occupancy_report(camera_id, camera)

    assert first["durationReady"] is False
    assert second["durationReady"] is True
    assert second["durationReadyZoneCount"] == 1
    assert second["maxLongestOccupiedSeconds"] == 6
    assert second["chairs"][0]["occupied"] is True
    assert second["chairs"][0]["longestOccupiedSeconds"] == 6
    assert second["chairs"][0]["durationReady"] is True
