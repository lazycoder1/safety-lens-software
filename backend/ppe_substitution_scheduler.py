"""Per-camera cadence for tracker-gated PPE specialist substitution."""

from __future__ import annotations

import math
import threading
from typing import Any


class PPESubstitutionScheduler:
    """Bound PPE duty and optionally replace a due primary slot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_selected_at: dict[str, float] = {}
        self._counters = {
            "cadence_suppressed_frames": 0,
            "selected_frames": 0,
            "confirmation_selected_frames": 0,
            "substituted_frames": 0,
            "additive_frames": 0,
        }

    @staticmethod
    def settings(cfg: dict) -> tuple[float, float, bool]:
        global_config = cfg.get("global") or {}
        try:
            target_fps = float(global_config.get("ppe_specialist_target_fps", 0.5))
        except (TypeError, ValueError):
            target_fps = 0.5
        if not math.isfinite(target_fps) or not 0.1 <= target_fps <= 2.0:
            target_fps = 0.5
        try:
            confirmation_fps = float(
                global_config.get("ppe_specialist_confirmation_fps", 1.0)
            )
        except (TypeError, ValueError):
            confirmation_fps = 1.0
        if not math.isfinite(confirmation_fps) or not 0.1 <= confirmation_fps <= 2.0:
            confirmation_fps = 1.0
        confirmation_fps = max(target_fps, confirmation_fps)
        substitution_enabled = (
            global_config.get("ppe_specialist_substitution_enabled") is True
        )
        return target_fps, confirmation_fps, substitution_enabled

    def consider(
        self,
        camera_id: str,
        cfg: dict,
        *,
        now: float,
        substitution_eligible: bool,
        confirmation_required: bool = False,
    ) -> tuple[bool, bool]:
        """Return ``(due, substitute)`` for an otherwise actionable PPE pass."""
        target_fps, confirmation_fps, substitution_enabled = self.settings(cfg)
        if confirmation_required:
            target_fps = confirmation_fps
        interval = 1.0 / target_fps
        with self._lock:
            last_selected = self._last_selected_at.get(camera_id)
            if last_selected is not None and now - last_selected < interval:
                self._counters["cadence_suppressed_frames"] += 1
                return False, False
            self._last_selected_at[camera_id] = now
            substitute = substitution_enabled and substitution_eligible
            self._counters["selected_frames"] += 1
            self._counters["confirmation_selected_frames"] += int(
                confirmation_required
            )
            self._counters[
                "substituted_frames" if substitute else "additive_frames"
            ] += 1
            return True, substitute

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tracked_cameras": len(self._last_selected_at),
                **self._counters,
            }

    def reset(self, camera_id: str | None = None) -> None:
        with self._lock:
            if camera_id is None:
                self._last_selected_at.clear()
                for key in self._counters:
                    self._counters[key] = 0
                return
            self._last_selected_at.pop(camera_id, None)


SCHEDULER = PPESubstitutionScheduler()
