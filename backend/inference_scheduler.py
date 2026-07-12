"""Deterministic camera inference phase scheduling."""

from __future__ import annotations

import math
import os
import time


INFERENCE_PHASE_EPOCH = time.monotonic()


def _phase_group_size() -> int:
    try:
        value = int(os.environ.get("SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE", "1"))
    except (TypeError, ValueError):
        value = 1
    return min(4, max(1, value))


def camera_phase_groups(cfg: dict) -> list[list[str]]:
    """Return the stable camera groups that share one inference phase."""
    camera_ids = sorted(
        str(configured_id)
        for configured_id, camera in (cfg.get("cameras") or {}).items()
        if not isinstance(camera, dict) or camera.get("enabled", True)
    )
    group_size = _phase_group_size()
    return [
        camera_ids[index : index + group_size]
        for index in range(0, len(camera_ids), group_size)
    ]


def camera_phase_offset(camera_id: str, cfg: dict, interval_seconds: float) -> float:
    """Spread enabled cameras evenly across one inference interval."""
    if interval_seconds <= 0:
        raise ValueError("Inference interval must be positive")
    groups = camera_phase_groups(cfg)
    camera_ids = [camera for group in groups for camera in group]
    if camera_id not in camera_ids:
        camera_ids.append(camera_id)
        camera_ids.sort()
        group_size = _phase_group_size()
        groups = [
            camera_ids[index : index + group_size]
            for index in range(0, len(camera_ids), group_size)
        ]
    group_count = len(groups)
    group_index = next(
        index for index, group in enumerate(groups) if camera_id in group
    )
    return group_index * interval_seconds / group_count


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
