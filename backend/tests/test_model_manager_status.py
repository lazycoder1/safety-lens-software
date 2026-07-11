import time

import model_manager
import state
from routers.cameras import _camera_runtime_status


def test_list_models_for_status_uses_short_timeout_and_failure_cache(monkeypatch):
    model_manager.clear_remote_model_status_cache()
    calls = []

    def fake_remote_get(path, *, timeout_seconds=None):
        calls.append((path, timeout_seconds))
        raise TimeoutError("model server hung")

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_settings", lambda: {"url": "http://models", "token": "", "timeout_seconds": 20})
    monkeypatch.setattr(model_manager, "_remote_get", fake_remote_get)

    first = model_manager.list_models_for_status()
    second = model_manager.list_models_for_status()

    assert calls == [("/api/models", model_manager.REMOTE_MODEL_STATUS_TIMEOUT_SECONDS)]
    assert first[0]["status"] == "remote_unavailable"
    assert second[0]["status"] == "remote_unavailable"
    assert "model server hung" in first[0]["error"]

    model_manager.clear_remote_model_status_cache()


def test_list_models_for_status_caches_successful_status(monkeypatch):
    model_manager.clear_remote_model_status_cache()
    calls = []
    payload = {
        "models": [
            {"model_key": "coco_primary", "status": "ready", "is_ready": True},
        ]
    }

    def fake_remote_get(path, *, timeout_seconds=None):
        calls.append((path, timeout_seconds))
        return payload

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_get", fake_remote_get)

    assert model_manager.list_models_for_status() == payload["models"]
    assert model_manager.list_models_for_status() == payload["models"]
    assert calls == [("/api/models", model_manager.REMOTE_MODEL_STATUS_TIMEOUT_SECONDS)]

    model_manager.clear_remote_model_status_cache()


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
