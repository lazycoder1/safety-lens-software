"""Tests for object lifecycle telemetry."""

import object_lifecycle_analytics


def _camera():
    return {
        "id": "object_cam",
        "name": "Object Camera",
        "zone": "Retail Floor",
        "capabilities": ["object_lifecycle"],
        "object_removal_after_seconds": 1.0,
        "object_dwell_after_seconds": 2.0,
        "object_event_linger_seconds": 10,
        "zones": [
            {
                "id": "bag_zone",
                "name": "Bag Watch Zone",
                "type": "object_watch",
                "analytics": "object_lifecycle",
                "classes": ["handbag", "suitcase", "umbrella"],
                "points": [[0.0, 0.4], [1.0, 0.4], [1.0, 1.0], [0.0, 1.0]],
            }
        ],
    }


def test_object_lifecycle_emits_removal_after_seen_object_disappears():
    camera_id = "object_cam"
    camera = _camera()
    object_lifecycle_analytics.reset_object_lifecycle_state(camera_id)

    present = [{"class": "handbag", "confidence": 0.82, "bbox": [100, 700, 560, 1210]}]
    absent = [{"class": "person", "confidence": 0.90, "bbox": [450, 0, 720, 1000]}]

    assert object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, present, 720, 1280, now=100.0) == []
    assert object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, absent, 720, 1280, now=100.5) == []
    events = object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, absent, 720, 1280, now=101.2)

    assert len(events) == 1
    assert events[0]["rule"] == "Object Removed"
    assert events[0]["zone"] == "Bag Watch Zone"
    assert "handbag" in events[0]["classes"]

    snapshot = object_lifecycle_analytics.get_object_lifecycle_snapshot(camera_id, camera)
    assert snapshot["removalCount"] == 1
    assert snapshot["watchZones"][0]["seenEver"] is True


def test_object_lifecycle_does_not_emit_without_prior_seen_object():
    camera_id = "object_cam_no_prior"
    camera = _camera()
    object_lifecycle_analytics.reset_object_lifecycle_state(camera_id)

    events = object_lifecycle_analytics.update_object_lifecycle(
        camera_id,
        camera,
        [{"class": "person", "confidence": 0.90, "bbox": [450, 0, 720, 1000]}],
        720,
        1280,
        now=100.0,
    )

    assert events == []


def test_object_lifecycle_emits_unattended_object_dwell_after_duration(monkeypatch):
    camera_id = "object_cam_dwell"
    camera = _camera()
    object_lifecycle_analytics.reset_object_lifecycle_state(camera_id)
    present = [{"class": "handbag", "confidence": 0.82, "bbox": [100, 700, 560, 1210]}]

    assert object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, present, 720, 1280, now=100.0) == []
    assert object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, present, 720, 1280, now=101.0) == []
    events = object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, present, 720, 1280, now=102.2)

    assert len(events) == 1
    assert events[0]["rule"] == "Unattended Object Dwell"
    assert "unattended object dwell" in events[0]["classes"]
    assert "handbag" in events[0]["classes"]
    assert events[0]["metadata"]["presentSeconds"] == 2.2

    monkeypatch.setattr(object_lifecycle_analytics.time, "time", lambda: 102.3)
    snapshot = object_lifecycle_analytics.get_object_lifecycle_snapshot(camera_id, camera)
    assert snapshot["dwellDetected"] is True
    assert snapshot["dwellCount"] == 1
    assert snapshot["watchZones"][0]["dwellReady"] is True


def test_object_lifecycle_dwell_event_is_one_shot_while_object_remains():
    camera_id = "object_cam_dwell_once"
    camera = _camera()
    object_lifecycle_analytics.reset_object_lifecycle_state(camera_id)
    present = [{"class": "handbag", "confidence": 0.82, "bbox": [100, 700, 560, 1210]}]

    object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, present, 720, 1280, now=100.0)
    first = object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, present, 720, 1280, now=102.1)
    second = object_lifecycle_analytics.update_object_lifecycle(camera_id, camera, present, 720, 1280, now=103.1)

    assert len(first) == 1
    assert second == []
