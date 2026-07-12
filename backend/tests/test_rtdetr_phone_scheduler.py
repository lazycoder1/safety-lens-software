import rtdetr_phone_scheduler
from rtdetr_phone_scheduler import (
    PrimaryPersonTracker,
    RTDETRPhoneSubstitutionScheduler,
)


def _person(x1=10, y1=10, x2=50, y2=100):
    return {
        "class": "person",
        "model_family": "coco_primary",
        "bbox": [x1, y1, x2, y2],
    }


def _cfg(camera_count=4, *, enabled=True):
    return {
        "cameras": {f"cam{index}": {"enabled": True} for index in range(camera_count)},
        "global": {
            "rtdetr_phone_substitution_enabled": enabled,
            "rtdetr_phone_target_fps": 1.0,
            "rtdetr_phone_person_track_ttl_seconds": 1.0,
        },
    }


def test_primary_person_tracker_requires_repeated_iou_matched_hits():
    tracker = PrimaryPersonTracker()
    tracker.update([_person()], now=1.0, ttl_seconds=1.0)
    assert tracker.has_stable_person(now=1.0, min_hits=2, ttl_seconds=1.0) is False

    tracker.update([_person(12, 10, 52, 100)], now=1.25, ttl_seconds=1.0)

    assert tracker.has_stable_person(now=1.25, min_hits=2, ttl_seconds=1.0) is True
    assert tracker.snapshot()[0]["hits"] == 2


def test_primary_person_tracker_expires_stale_context():
    tracker = PrimaryPersonTracker()
    tracker.update([_person()], now=1.0, ttl_seconds=1.0)
    tracker.update([_person()], now=1.2, ttl_seconds=1.0)

    assert tracker.has_stable_person(now=2.21, min_hits=2, ttl_seconds=1.0) is False


def test_scheduler_selects_one_phase_aligned_pair_after_warmup(monkeypatch):
    monkeypatch.setattr(
        rtdetr_phone_scheduler.inference_scheduler,
        "camera_phase_groups",
        lambda _cfg: [["cam0", "cam1", "cam2", "cam3"]],
    )
    scheduler = RTDETRPhoneSubstitutionScheduler()
    cfg = _cfg()
    for camera_id in cfg["cameras"]:
        assert (
            scheduler.consider(
                camera_id,
                cfg,
                now=0.0,
                stable_person=True,
            )
            is False
        )
    for camera_id in cfg["cameras"]:
        scheduler.consider(camera_id, cfg, now=1.5, stable_person=True)

    decisions = {
        camera_id: scheduler.consider(
            camera_id,
            cfg,
            now=2.0,
            stable_person=True,
        )
        for camera_id in cfg["cameras"]
    }

    assert decisions == {"cam0": True, "cam1": True, "cam2": False, "cam3": False}
    assert scheduler.stats()["selected_pairs"] == 1
    assert scheduler.stats()["selected_frames"] == 2


def test_scheduler_round_robins_pairs_without_crossing_phase_groups(monkeypatch):
    monkeypatch.setattr(
        rtdetr_phone_scheduler.inference_scheduler,
        "camera_phase_groups",
        lambda _cfg: [["cam0", "cam1"], ["cam2", "cam3"]],
    )
    scheduler = RTDETRPhoneSubstitutionScheduler()
    cfg = _cfg()
    for camera_id in cfg["cameras"]:
        scheduler.consider(camera_id, cfg, now=0.0, stable_person=True)
    for camera_id in cfg["cameras"]:
        scheduler.consider(camera_id, cfg, now=1.5, stable_person=True)
    first = [
        camera_id
        for camera_id in cfg["cameras"]
        if scheduler.consider(camera_id, cfg, now=2.0, stable_person=True)
    ]
    for camera_id in cfg["cameras"]:
        scheduler.consider(camera_id, cfg, now=3.5, stable_person=True)
    second = [
        camera_id
        for camera_id in cfg["cameras"]
        if scheduler.consider(camera_id, cfg, now=4.0, stable_person=True)
    ]

    assert first == ["cam0", "cam1"]
    assert second == ["cam2", "cam3"]


def test_scheduler_is_default_off_and_requires_two_stable_contexts():
    scheduler = RTDETRPhoneSubstitutionScheduler()
    disabled = _cfg(enabled=False)
    assert (
        scheduler.consider(
            "cam0",
            disabled,
            now=10.0,
            stable_person=True,
        )
        is False
    )

    enabled = _cfg(camera_count=3)
    scheduler.consider("cam0", enabled, now=0.0, stable_person=True)
    assert (
        scheduler.consider(
            "cam0",
            enabled,
            now=2.0,
            stable_person=True,
        )
        is False
    )


def test_scheduler_allows_singleton_for_small_live_deployment(monkeypatch):
    monkeypatch.setattr(
        rtdetr_phone_scheduler.inference_scheduler,
        "camera_phase_groups",
        lambda _cfg: [["cam0", "cam1"]],
    )
    scheduler = RTDETRPhoneSubstitutionScheduler()
    cfg = _cfg(camera_count=2)
    scheduler.consider("cam0", cfg, now=0.0, stable_person=True)

    assert scheduler.consider(
        "cam0",
        cfg,
        now=1.0,
        stable_person=True,
    ) is True
    assert scheduler.stats()["selected_singletons"] == 1
