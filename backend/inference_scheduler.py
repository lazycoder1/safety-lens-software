"""Deterministic camera inference phase scheduling."""

from __future__ import annotations

import math
import time


INFERENCE_PHASE_EPOCH = time.monotonic()


def camera_phase_offset(camera_id: str, cfg: dict, interval_seconds: float) -> float:
    """Spread enabled cameras evenly across one inference interval."""
    if interval_seconds <= 0:
        raise ValueError("Inference interval must be positive")
    camera_ids = sorted(
        str(configured_id)
        for configured_id, camera in (cfg.get("cameras") or {}).items()
        if not isinstance(camera, dict) or camera.get("enabled", True)
    )
    if camera_id not in camera_ids:
        camera_ids.append(camera_id)
        camera_ids.sort()
    return camera_ids.index(camera_id) * interval_seconds / len(camera_ids)


def next_inference_slot(
    camera_id: str,
    cfg: dict,
    interval_seconds: float,
    *,
    now: float | None = None,
    epoch: float = INFERENCE_PHASE_EPOCH,
) -> float:
    """Return this camera's next slot on the process-wide monotonic grid."""
    now = time.monotonic() if now is None else now
    phase = camera_phase_offset(camera_id, cfg, interval_seconds)
    first_slot = epoch + phase
    if now <= first_slot:
        return first_slot
    cycles = math.ceil((now - first_slot) / interval_seconds)
    return first_slot + cycles * interval_seconds
