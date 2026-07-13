from ppe_substitution_scheduler import PPESubstitutionScheduler


def _config(*, enabled=True, target_fps=0.5):
    return {
        "global": {
            "ppe_specialist_target_fps": target_fps,
            "ppe_specialist_substitution_enabled": enabled,
        }
    }


def test_scheduler_selects_immediately_then_enforces_per_camera_cadence():
    scheduler = PPESubstitutionScheduler()
    cfg = _config()

    assert scheduler.consider(
        "cam1", cfg, now=10.0, substitution_eligible=True
    ) == (True, True)
    assert scheduler.consider(
        "cam1", cfg, now=11.9, substitution_eligible=True
    ) == (False, False)
    assert scheduler.consider(
        "cam1", cfg, now=12.0, substitution_eligible=True
    ) == (True, True)


def test_scheduler_keeps_due_pass_additive_until_tracking_is_stable():
    scheduler = PPESubstitutionScheduler()

    assert scheduler.consider(
        "cam1",
        _config(),
        now=10.0,
        substitution_eligible=False,
    ) == (True, False)


def test_scheduler_can_bound_duty_without_enabling_substitution():
    scheduler = PPESubstitutionScheduler()
    cfg = _config(enabled=False, target_fps=1.0)

    assert scheduler.consider(
        "cam1", cfg, now=10.0, substitution_eligible=True
    ) == (True, False)
    assert scheduler.consider(
        "cam1", cfg, now=10.5, substitution_eligible=True
    ) == (False, False)
    assert scheduler.consider(
        "cam1", cfg, now=11.0, substitution_eligible=True
    ) == (True, False)


def test_scheduler_tracks_each_camera_independently_and_reports_stats():
    scheduler = PPESubstitutionScheduler()
    cfg = _config()

    scheduler.consider("cam1", cfg, now=10.0, substitution_eligible=True)
    scheduler.consider("cam2", cfg, now=10.1, substitution_eligible=False)
    scheduler.consider("cam1", cfg, now=10.2, substitution_eligible=True)

    assert scheduler.stats() == {
        "tracked_cameras": 2,
        "cadence_suppressed_frames": 1,
        "selected_frames": 2,
        "substituted_frames": 1,
        "additive_frames": 1,
    }


def test_invalid_target_uses_safe_half_fps_default():
    scheduler = PPESubstitutionScheduler()
    cfg = _config(target_fps=99)

    assert scheduler.consider(
        "cam1", cfg, now=10.0, substitution_eligible=True
    ) == (True, True)
    assert scheduler.consider(
        "cam1", cfg, now=11.0, substitution_eligible=True
    ) == (False, False)
    assert scheduler.consider(
        "cam1", cfg, now=12.0, substitution_eligible=True
    ) == (True, True)
