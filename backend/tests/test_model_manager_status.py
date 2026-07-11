import time

import model_manager
import state
from routers.cameras import _camera_runtime_status


def test_list_models_for_status_keeps_compatibility_wrapper(monkeypatch):
    payload = [{"model_key": "coco_primary", "status": "ready", "is_ready": True}]
    calls = []

    def list_models(**kwargs):
        calls.append(kwargs)
        return payload

    monkeypatch.setattr(model_manager, "list_models", list_models)

    assert model_manager.list_models_for_status() == payload
    assert calls == [{
        "timeout_seconds": model_manager.REMOTE_MODEL_STATUS_TIMEOUT_SECONDS,
        "allow_cached": True,
    }]


def test_running_camera_status_does_not_block_on_model_readiness(monkeypatch):
    cam_id = "cam_status"
    state.camera_threads[cam_id] = (object(), object())
    state.camera_frames[cam_id] = object()
    state.camera_frame_updated_at[cam_id] = time.time()

    def fail_if_called(_model_keys):
        raise AssertionError("fresh running camera should not check model readiness")

    monkeypatch.setattr(model_manager, "missing_model_keys", fail_if_called)

    try:
        status = _camera_runtime_status(
            cam_id,
            {"enabled": True, "execution_plan": {"required_model_keys": ["coco_primary"]}},
            {"global": {}, "cameras": {}},
        )
        assert status == "running"
    finally:
        state.camera_threads.pop(cam_id, None)
        state.camera_frames.pop(cam_id, None)
        state.camera_frame_updated_at.pop(cam_id, None)
