"""Feature-gated per-camera adaptive inference decisions.

The controller is intentionally independent of camera workers and model code.  It
can therefore run in ``shadow`` mode first: the existing configured cadence stays
authoritative while the controller records the lower-rate decision it would have
made.  Urgent signals never disappear behind the rate controller in ``active``
mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


AdaptiveMode = Literal["off", "shadow", "active"]
ActivityState = Literal["quiet", "uncertain", "active"]

VALID_MODES = frozenset({"off", "shadow", "active"})
URGENT_REASON_NEW_ENTRY = "new_entry"
URGENT_REASON_LOW_TRACKER_CONFIDENCE = "low_tracker_confidence"
URGENT_REASON_ZONE_ENTRY = "zone_entry"
URGENT_REASON_POSSIBLE_VIOLATION = "possible_violation"
URGENT_REASON_MAX_KEYFRAME_AGE = "max_keyframe_age"
URGENT_REASON_ALERT_CONFIRMATION = "alert_confirmation"
URGENT_REASON_TRACKER_REDETECT = "tracker_force_redetect"
_URGENT_REASON_ORDER = (
    URGENT_REASON_NEW_ENTRY,
    URGENT_REASON_LOW_TRACKER_CONFIDENCE,
    URGENT_REASON_ZONE_ENTRY,
    URGENT_REASON_POSSIBLE_VIOLATION,
    URGENT_REASON_TRACKER_REDETECT,
    URGENT_REASON_MAX_KEYFRAME_AGE,
    URGENT_REASON_ALERT_CONFIRMATION,
)


@dataclass(frozen=True)
class AdaptiveSignals:
    """One camera observation used to update adaptive state.

    ``motion_score`` is expected to be a fraction in the inclusive ``0..1``
    range.  ``tracker_confidence=None`` means no tracker judgement was made; an
    invalid or non-finite supplied confidence fails safe as low confidence.
    """

    motion_score: float = 0.0
    person_present: bool = False
    tracker_confidence: float | None = None
    new_entry: bool = False
    zone_entry: bool = False
    possible_violation: bool = False
    alert_confirmation: bool = False
    force_redetect: bool = False


@dataclass(frozen=True)
class AdaptiveDecision:
    """A baseline decision and its adaptive counterfactual.

    ``submit_now`` is safe to apply for the configured mode.  In ``off`` and
    ``shadow`` modes it follows the configured baseline cadence.  In ``active``
    mode it follows ``adaptive_due``.  This makes shadow telemetry incapable of
    changing production inference by accident.
    """

    mode: AdaptiveMode
    state: ActivityState
    target_fps: float
    submit_now: bool
    baseline_due: bool
    adaptive_due: bool
    urgent_reasons: tuple[str, ...]

    @property
    def urgent(self) -> bool:
        return bool(self.urgent_reasons)

    @property
    def interval_seconds(self) -> float:
        return 1.0 / self.target_fps


def _finite_positive(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


class AdaptiveInferenceController:
    """Three-state, hysteretic inference-rate controller for one camera.

    Quiet cameras target one FPS, uncertain cameras target two FPS, and active
    cameras target the configured FPS.  Neither quiet nor uncertain operation is
    allowed to exceed the configured ceiling.

    ``decide`` never consumes a cadence slot or urgency edge. Call
    ``record_dispatch`` only after work is accepted, and ``record_inference``
    when it completes; this keeps rejected offers from losing re-detection.
    """

    def __init__(
        self,
        configured_fps: float,
        *,
        mode: AdaptiveMode = "off",
        quiet_motion_threshold: float = 0.001,
        active_motion_threshold: float = 0.02,
        tracker_confidence_threshold: float = 0.55,
        max_keyframe_age_seconds: float = 1.0,
        quiet_enter_observations: int = 4,
        active_enter_observations: int = 2,
        active_exit_observations: int = 3,
    ) -> None:
        self.configured_fps = _finite_positive(configured_fps, "configured_fps")
        self.mode = self._validated_mode(mode)
        self.quiet_motion_threshold = self._fraction(
            quiet_motion_threshold,
            "quiet_motion_threshold",
        )
        self.active_motion_threshold = self._fraction(
            active_motion_threshold,
            "active_motion_threshold",
        )
        if self.active_motion_threshold <= self.quiet_motion_threshold:
            raise ValueError(
                "active_motion_threshold must be greater than quiet_motion_threshold"
            )
        self.tracker_confidence_threshold = self._fraction(
            tracker_confidence_threshold,
            "tracker_confidence_threshold",
        )
        self.max_keyframe_age_seconds = _finite_positive(
            max_keyframe_age_seconds,
            "max_keyframe_age_seconds",
        )
        self.quiet_enter_observations = _positive_int(
            quiet_enter_observations,
            "quiet_enter_observations",
        )
        self.active_enter_observations = _positive_int(
            active_enter_observations,
            "active_enter_observations",
        )
        self.active_exit_observations = _positive_int(
            active_exit_observations,
            "active_exit_observations",
        )

        self._state: ActivityState = "uncertain"
        self._quiet_streak = 0
        self._active_streak = 0
        self._inactive_streak = 0
        self._last_observed_at: float | None = None
        self._last_actual_inference_at: float | None = None
        self._last_adaptive_dispatch_at: float | None = None
        self._last_keyframe_at: float | None = None
        self._previous_urgent_reasons: frozenset[str] = frozenset()
        self._pending_urgent_reasons: set[str] = set()

    @staticmethod
    def _validated_mode(mode: str) -> AdaptiveMode:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        return mode  # type: ignore[return-value]

    @staticmethod
    def _fraction(value: float, name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be between 0 and 1") from exc
        if not math.isfinite(parsed) or not 0 <= parsed <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
        return parsed

    @staticmethod
    def _timestamp(value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("now must be a finite monotonic timestamp") from exc
        if not math.isfinite(parsed):
            raise ValueError("now must be a finite monotonic timestamp")
        return parsed

    @property
    def state(self) -> ActivityState:
        return "active" if self.mode == "off" else self._state

    @property
    def last_keyframe_at(self) -> float | None:
        return self._last_keyframe_at

    def set_mode(self, mode: AdaptiveMode) -> None:
        validated = self._validated_mode(mode)
        if validated != self.mode:
            # Enabling a new policy starts with a fresh admission opportunity;
            # a shadow-mode counterfactual must not delay the first active frame.
            self._last_adaptive_dispatch_at = None
        self.mode = validated
        self._previous_urgent_reasons = frozenset()
        self._pending_urgent_reasons.clear()

    def reset(self) -> None:
        """Forget cadence and hysteresis state for a reconnect or source change."""

        self._state = "uncertain"
        self._quiet_streak = 0
        self._active_streak = 0
        self._inactive_streak = 0
        self._last_observed_at = None
        self._last_actual_inference_at = None
        self._last_adaptive_dispatch_at = None
        self._last_keyframe_at = None
        self._previous_urgent_reasons = frozenset()
        self._pending_urgent_reasons.clear()

    def record_dispatch(self, now: float) -> None:
        """Consume one adaptive slot only after inference work is accepted."""

        timestamp = self._timestamp(now)
        if (
            self._last_adaptive_dispatch_at is not None
            and timestamp < self._last_adaptive_dispatch_at
        ):
            raise ValueError("dispatch timestamps must be monotonic")
        self._last_adaptive_dispatch_at = timestamp
        self._pending_urgent_reasons.clear()

    def record_inference(
        self,
        now: float,
        *,
        keyframe: bool = True,
        keyframe_at: float | None = None,
    ) -> None:
        """Record completion while anchoring keyframe freshness to capture time."""

        timestamp = self._timestamp(now)
        if (
            self._last_actual_inference_at is not None
            and timestamp < self._last_actual_inference_at
        ):
            raise ValueError("inference timestamps must be monotonic")
        self._last_actual_inference_at = timestamp
        if keyframe:
            captured_at = (
                timestamp if keyframe_at is None else self._timestamp(keyframe_at)
            )
            if captured_at > timestamp:
                raise ValueError("keyframe capture cannot follow inference completion")
            if (
                self._last_keyframe_at is not None
                and captured_at < self._last_keyframe_at
            ):
                raise ValueError("keyframe timestamps must be monotonic")
            self._last_keyframe_at = captured_at

    def record_keyframe(self, now: float) -> None:
        self.record_inference(now, keyframe=True)

    @staticmethod
    def _due(last_at: float | None, now: float, fps: float) -> bool:
        return last_at is None or now - last_at >= (1.0 / fps) - 1e-9

    def _target_fps(self) -> float:
        if self.mode == "off" or self._state == "active":
            return self.configured_fps
        if self._state == "uncertain":
            return min(2.0, self.configured_fps)
        return min(1.0, self.configured_fps)

    def _urgent_reasons(
        self,
        now: float,
        signals: AdaptiveSignals,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if signals.new_entry:
            reasons.append(URGENT_REASON_NEW_ENTRY)
        confidence = signals.tracker_confidence
        if confidence is not None:
            try:
                parsed_confidence = float(confidence)
            except (TypeError, ValueError):
                parsed_confidence = float("nan")
            if (
                not math.isfinite(parsed_confidence)
                or parsed_confidence < self.tracker_confidence_threshold
            ):
                reasons.append(URGENT_REASON_LOW_TRACKER_CONFIDENCE)
        if signals.zone_entry:
            reasons.append(URGENT_REASON_ZONE_ENTRY)
        if signals.possible_violation:
            reasons.append(URGENT_REASON_POSSIBLE_VIOLATION)
        if signals.force_redetect:
            reasons.append(URGENT_REASON_TRACKER_REDETECT)
        if (
            self._last_keyframe_at is None
            or now - self._last_keyframe_at >= self.max_keyframe_age_seconds
        ):
            reasons.append(URGENT_REASON_MAX_KEYFRAME_AGE)
        if signals.alert_confirmation:
            reasons.append(URGENT_REASON_ALERT_CONFIRMATION)
        return tuple(reasons)

    def _observation_class(self, signals: AdaptiveSignals) -> ActivityState:
        try:
            motion_score = float(signals.motion_score)
        except (TypeError, ValueError):
            motion_score = 1.0
        if not math.isfinite(motion_score):
            motion_score = 1.0
        motion_score = min(1.0, max(0.0, motion_score))
        if signals.person_present or motion_score >= self.active_motion_threshold:
            return "active"
        if motion_score <= self.quiet_motion_threshold:
            return "quiet"
        return "uncertain"

    def _advance_hysteresis(
        self,
        observation: ActivityState,
        *,
        urgent: bool,
    ) -> None:
        if urgent:
            self._state = "active"
            self._quiet_streak = 0
            self._active_streak = 0
            self._inactive_streak = 0
            return

        if self._state == "active":
            if observation == "active":
                self._inactive_streak = 0
                self._quiet_streak = 0
                return
            self._inactive_streak += 1
            if self._inactive_streak >= self.active_exit_observations:
                self._state = "uncertain"
                self._inactive_streak = 0
                self._quiet_streak = 1 if observation == "quiet" else 0
            return

        if self._state == "quiet":
            if observation == "quiet":
                self._active_streak = 0
                return
            # Leave the one-FPS state immediately, but require hysteresis before
            # entering the configured active rate.
            self._state = "uncertain"
            self._quiet_streak = 0
            self._active_streak = 1 if observation == "active" else 0
            if self._active_streak >= self.active_enter_observations:
                self._state = "active"
                self._active_streak = 0
            return

        # uncertain
        if observation == "active":
            self._active_streak += 1
            self._quiet_streak = 0
            if self._active_streak >= self.active_enter_observations:
                self._state = "active"
                self._active_streak = 0
                self._inactive_streak = 0
            return
        if observation == "quiet":
            self._quiet_streak += 1
            self._active_streak = 0
            if self._quiet_streak >= self.quiet_enter_observations:
                self._state = "quiet"
                self._quiet_streak = 0
            return
        self._quiet_streak = 0
        self._active_streak = 0

    def decide(
        self,
        now: float,
        signals: AdaptiveSignals | None = None,
    ) -> AdaptiveDecision:
        """Return a safe decision for one captured frame."""

        timestamp = self._timestamp(now)
        if self._last_observed_at is not None and timestamp < self._last_observed_at:
            raise ValueError("observation timestamps must be monotonic")
        self._last_observed_at = timestamp
        signals = signals or AdaptiveSignals()

        observed_urgent_reasons = self._urgent_reasons(timestamp, signals)
        urgent_set = frozenset(observed_urgent_reasons)
        self._pending_urgent_reasons.update(
            urgent_set - self._previous_urgent_reasons
        )
        self._previous_urgent_reasons = urgent_set
        effective_urgent = urgent_set | self._pending_urgent_reasons
        urgent_reasons = tuple(
            reason for reason in _URGENT_REASON_ORDER if reason in effective_urgent
        )

        if self.mode != "off":
            self._advance_hysteresis(
                self._observation_class(signals),
                urgent=bool(urgent_reasons),
            )

        target_fps = self._target_fps()
        baseline_due = self._due(
            self._last_actual_inference_at,
            timestamp,
            self.configured_fps,
        )
        if self.mode == "off":
            adaptive_due = baseline_due
        else:
            adaptive_due = bool(self._pending_urgent_reasons) or self._due(
                self._last_adaptive_dispatch_at,
                timestamp,
                target_fps,
            )

        submit_now = adaptive_due if self.mode == "active" else baseline_due
        return AdaptiveDecision(
            mode=self.mode,
            state=self.state,
            target_fps=target_fps,
            submit_now=submit_now,
            baseline_due=baseline_due,
            adaptive_due=adaptive_due,
            urgent_reasons=urgent_reasons,
        )
