import sys
from types import SimpleNamespace

import numpy as np

import config_manager
import model_manager
from tensorrt_engine import build_manifest, manifest_path, write_manifest


def _write_valid_artifacts(tmp_path):
    source = tmp_path / "model.pt"
    engine = tmp_path / "model.engine"
    source.write_bytes(b"pytorch-model")
    engine.write_bytes(b"tensorrt-engine")
    write_manifest(
        manifest_path(engine),
        build_manifest(
            source_path=source,
            engine_path=engine,
            imgsz=960,
            precision="fp16",
            task="detect",
        ),
    )
    return source, engine


def test_coco_runtime_uses_explicit_valid_engine(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path)
    monkeypatch.setenv("SAFETYLENS_COCO_TENSORRT_ENGINE", str(engine))

    runtime_path, backend, fixed_imgsz, error = model_manager._configured_runtime_path(
        "coco_primary",
        source,
    )

    assert runtime_path == engine
    assert backend == "tensorrt"
    assert fixed_imgsz == 960
    assert error is None


def test_coco_runtime_rejects_engine_for_different_source(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path)
    source.write_bytes(b"new-model-release")
    monkeypatch.setenv("SAFETYLENS_COCO_TENSORRT_ENGINE", str(engine))

    runtime_path, backend, fixed_imgsz, error = model_manager._configured_runtime_path(
        "coco_primary",
        source,
    )

    assert runtime_path == source
    assert backend == "pytorch"
    assert fixed_imgsz is None
    assert "source-model hash" in error


def test_fixed_shape_engine_overrides_requested_image_size():
    class FakeHandle:
        def __init__(self):
            self.calls = []

        def predict(self, frame, **kwargs):
            self.calls.append((frame.shape, kwargs))
            return []

    handle = FakeHandle()
    runtime = model_manager._new_model_runtime()
    runtime.update(handle=handle, runtime_backend="tensorrt", fixed_imgsz=960, warmed=True)

    model_manager._predict_with_runtime(
        "coco_primary",
        runtime,
        np.zeros((32, 32, 3), dtype=np.uint8),
        conf=0.4,
        device="cuda",
        imgsz=640,
    )

    assert handle.calls[0][1]["imgsz"] == 960


def test_tensorrt_load_failure_uses_pytorch(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path)
    calls = []
    fallback_handle = object()

    def fake_yolo(path, task=None):
        calls.append((path, task))
        if path == str(engine):
            raise RuntimeError("incompatible engine")
        assert path == str(source)
        return fallback_handle

    runtime = model_manager._new_model_runtime()
    monkeypatch.setenv("SAFETYLENS_COCO_TENSORRT_ENGINE", str(engine))
    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "coco_primary", runtime)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.setattr(config_manager, "get_config", lambda: {"global": {"device": "cpu"}})

    model_manager._load_runtime("coco_primary", source)

    assert calls == [(str(engine), "detect"), (str(source), None)]
    assert runtime["handle"] is fallback_handle
    assert runtime["runtime_backend"] == "pytorch_fallback"
    assert runtime["runtime_path"] == str(source)
    assert runtime["runtime_fallback_error"] == "TensorRT load failed: incompatible engine"
    assert runtime["fallback_path"] is None
    assert runtime["fixed_imgsz"] is None


def test_tensorrt_prediction_failure_retries_once_with_pytorch(monkeypatch):
    class FailingHandle:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("engine execution failed")

    class FallbackHandle:
        def __init__(self):
            self.calls = []

        def predict(self, _frame, **kwargs):
            self.calls.append(kwargs)
            return ["fallback-result"]

    runtime = model_manager._new_model_runtime()
    runtime.update(
        handle=FailingHandle(),
        runtime_backend="tensorrt",
        fallback_path="/models/model.pt",
        fixed_imgsz=960,
        warmed=True,
    )
    fallback = FallbackHandle()

    def activate(active_runtime, error):
        assert str(error) == "engine execution failed"
        active_runtime.update(
            handle=fallback,
            runtime_backend="pytorch_fallback",
            fallback_path=None,
            fixed_imgsz=None,
            warmed=True,
        )

    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "coco_primary", runtime)
    monkeypatch.setattr(model_manager, "_activate_pytorch_fallback", activate)

    result = model_manager.predict(
        "coco_primary",
        np.zeros((32, 32, 3), dtype=np.uint8),
        conf=0.4,
        device="cuda",
        imgsz=640,
    )

    assert result == ["fallback-result"]
    assert fallback.calls[0]["imgsz"] == 640


def test_pytorch_fallback_releases_engine_and_updates_runtime(monkeypatch):
    fallback_handle = object()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=lambda path: (path, fallback_handle)),
    )
    runtime = {
        "handle": object(),
        "fallback_path": "/models/model.pt",
        "runtime_backend": "tensorrt",
        "runtime_path": "/models/model.engine",
        "fixed_imgsz": 960,
        "warmed": True,
    }

    model_manager._activate_pytorch_fallback(runtime, RuntimeError("bad engine"))

    assert runtime["handle"] == ("/models/model.pt", fallback_handle)
    assert runtime["runtime_backend"] == "pytorch_fallback"
    assert runtime["runtime_path"] == "/models/model.pt"
    assert runtime["runtime_fallback_error"] == "bad engine"
    assert runtime["fallback_path"] is None
    assert runtime["fixed_imgsz"] is None
    assert runtime["warmed"] is False
