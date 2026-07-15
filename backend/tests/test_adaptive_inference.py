import pytest

from adaptive_inference import (
    AdaptiveInferenceController,
    AdaptiveSignals,
    URGENT_REASON_ALERT_CONFIRMATION,
    URGENT_REASON_LOW_TRACKER_CONFIDENCE,
    URGENT_REASON_MAX_KEYFRAME_AGE,
    URGENT_REASON_NEW_ENTRY,
    URGENT_REASON_POSSIBLE_VIOLATION,
    URGENT_REASON_TRACKER_REDETECT,
    URGENT_REASON_ZONE_ENTRY,
)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        AdaptiveInferenceController(0)
    with pytest.raises(ValueError):
        AdaptiveInferenceController(4, mode="enabled")
    with pytest.raises(ValueError):
        AdaptiveInferenceController(
            4,
            quiet_motion_threshold=0.2,
            active_motion_threshold=0.1,
        )


def test_off_mode_preserves_configured_baseline_and_reports_active_state():
    controller = AdaptiveInferenceController(4, mode="off")

    first = controller.decide(0.0)
    controller.record_inference(0.0)
    early = controller.decide(0.2)
    due = controller.decide(0.25)

    assert first.submit_now is True
    assert first.target_fps == 4
    assert first.state == "active"
    assert early.submit_now is False
    assert due.submit_now is True
    assert due.baseline_due is True
    assert due.adaptive_due is True


def test_shadow_mode_never_changes_baseline_admission():
    controller = AdaptiveInferenceController(
        4,
        mode="shadow",
        quiet_enter_observations=2,
        max_keyframe_age_seconds=10,
    )
    controller.record_keyframe(0.0)

    controller.decide(0.0, AdaptiveSignals(motion_score=0.0))
    controller.record_dispatch(0.0)
    quiet = controller.decide(0.1, AdaptiveSignals(motion_score=0.0))
    baseline_only = controller.decide(0.25, AdaptiveSignals(motion_score=0.0))

    assert quiet.state == "quiet"
    assert quiet.target_fps == 1
    assert baseline_only.baseline_due is True
    assert baseline_only.adaptive_due is False
    assert baseline_only.submit_now is True


def test_quiet_active_and_exit_hysteresis():
    controller = AdaptiveInferenceController(
        5,
        mode="active",
        quiet_enter_observations=2,
        active_enter_observations=2,
        active_exit_observations=2,
        max_keyframe_age_seconds=100,
    )
    controller.record_keyframe(0.0)

    assert controller.decide(0.0, AdaptiveSignals()).state == "uncertain"
    assert controller.decide(0.1, AdaptiveSignals()).state == "quiet"
    assert (
        controller.decide(0.2, AdaptiveSignals(motion_score=0.01)).state == "uncertain"
    )
    assert (
        controller.decide(0.3, AdaptiveSignals(person_present=True)).state
        == "uncertain"
    )
    assert (
        controller.decide(0.4, AdaptiveSignals(person_present=True)).state == "active"
    )
    assert controller.decide(0.5, AdaptiveSignals()).state == "active"
    assert controller.decide(0.6, AdaptiveSignals()).state == "uncertain"
    assert controller.decide(0.7, AdaptiveSignals()).state == "quiet"


def test_all_urgent_signals_bypass_rate_limit_once():
    controller = AdaptiveInferenceController(
        1,
        mode="active",
        max_keyframe_age_seconds=100,
    )
    controller.record_inference(0.0)

    decision = controller.decide(
        0.1,
        AdaptiveSignals(
            tracker_confidence=0.1,
            new_entry=True,
            zone_entry=True,
            possible_violation=True,
            alert_confirmation=True,
        ),
    )

    assert decision.baseline_due is False
    assert decision.adaptive_due is True
    assert decision.submit_now is True
    assert decision.urgent_reasons == (
        URGENT_REASON_NEW_ENTRY,
        URGENT_REASON_LOW_TRACKER_CONFIDENCE,
        URGENT_REASON_ZONE_ENTRY,
        URGENT_REASON_POSSIBLE_VIOLATION,
        URGENT_REASON_ALERT_CONFIRMATION,
    )


def test_persistent_urgent_signal_does_not_submit_every_capture_frame():
    controller = AdaptiveInferenceController(
        4,
        mode="active",
        max_keyframe_age_seconds=1,
    )
    controller.record_keyframe(0.0)
    controller.decide(0.0)
    controller.record_dispatch(0.0)

    first_stale = controller.decide(1.0)
    controller.record_dispatch(1.0)
    repeated = controller.decide(1.05)
    rate_due = controller.decide(1.25)

    assert URGENT_REASON_MAX_KEYFRAME_AGE in first_stale.urgent_reasons
    assert first_stale.submit_now is True
    assert repeated.urgent is True
    assert repeated.submit_now is False
    assert rate_due.submit_now is True


def test_tracker_hard_fallback_requests_immediate_keyframe():
    controller = AdaptiveInferenceController(
        1,
        mode="active",
        max_keyframe_age_seconds=100,
    )
    controller.record_keyframe(0.0)
    controller.decide(0.0)
    controller.record_dispatch(0.0)

    decision = controller.decide(
        0.1,
        AdaptiveSignals(force_redetect=True),
    )

    assert decision.baseline_due is False
    assert decision.submit_now is True
    assert URGENT_REASON_TRACKER_REDETECT in decision.urgent_reasons


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), "bad"])
def test_invalid_tracker_confidence_fails_safe(confidence):
    controller = AdaptiveInferenceController(
        4,
        mode="active",
        max_keyframe_age_seconds=100,
    )
    controller.record_keyframe(0.0)

    decision = controller.decide(
        0.1,
        AdaptiveSignals(tracker_confidence=confidence),
    )

    assert URGENT_REASON_LOW_TRACKER_CONFIDENCE in decision.urgent_reasons


def test_configured_rate_is_always_the_ceiling_and_timestamps_are_monotonic():
    controller = AdaptiveInferenceController(
        0.5,
        mode="active",
        quiet_enter_observations=1,
        max_keyframe_age_seconds=100,
    )
    controller.record_keyframe(1.0)

    decision = controller.decide(1.0)

    assert decision.state == "quiet"
    assert decision.target_fps == 0.5
    with pytest.raises(ValueError):
        controller.decide(0.9)


def test_enabling_active_mode_is_not_delayed_by_shadow_clock():
    controller = AdaptiveInferenceController(
        1,
        mode="shadow",
        max_keyframe_age_seconds=100,
    )
    controller.record_keyframe(0.0)
    controller.decide(0.1)
    controller.record_dispatch(0.1)

    controller.set_mode("active")
    decision = controller.decide(0.2)

    assert decision.adaptive_due is True
    assert decision.submit_now is True


def test_rejected_urgent_offer_is_retried_until_dispatch_is_recorded():
    controller = AdaptiveInferenceController(
        1,
        mode="active",
        max_keyframe_age_seconds=100,
    )
    controller.record_keyframe(0.0)
    controller.record_dispatch(0.0)

    rejected = controller.decide(0.1, AdaptiveSignals(new_entry=True))
    retry = controller.decide(0.11, AdaptiveSignals())

    assert rejected.submit_now is True
    assert retry.submit_now is True
    assert URGENT_REASON_NEW_ENTRY in retry.urgent_reasons

    controller.record_dispatch(0.11)
    consumed = controller.decide(0.12, AdaptiveSignals())
    assert consumed.submit_now is False


def test_rejected_cadence_slot_does_not_create_a_phantom_dispatch():
    controller = AdaptiveInferenceController(
        2,
        mode="active",
        max_keyframe_age_seconds=100,
    )
    controller.record_keyframe(0.0)
    controller.record_dispatch(0.0)

    rejected = controller.decide(0.5)
    retry = controller.decide(0.51)

    assert rejected.submit_now is True
    assert retry.submit_now is True
    controller.record_dispatch(0.51)
    assert controller.decide(0.52).submit_now is False
