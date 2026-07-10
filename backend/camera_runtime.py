"""Camera worker liveness and public runtime-status derivation."""

from __future__ import annotations

import state


def camera_worker_running(camera_id: str) -> bool:
    """Return whether the registered camera worker is actually alive."""
    worker = state.camera_threads.get(camera_id)
    return bool(worker and worker[0].is_alive())


def derive_camera_runtime_status(
    camera_id: str,
    camera: dict,
    *,
    missing_models: bool = False,
) -> str:
    """Derive a truthful status without erasing reconnect/stop transitions."""
    if missing_models:
        return "awaiting_model_install"
    if not camera.get("enabled", True):
        return "offline"
    if not camera_worker_running(camera_id):
        return "offline"
    if state.camera_frames.get(camera_id) is not None:
        return "running"

    stored_status = state.camera_runtime_status.get(camera_id)
    if stored_status in {"starting", "reconnecting", "stopping"}:
        return stored_status
    return "starting"
