import sys
from types import SimpleNamespace

import numpy as np

import config_manager
import model_manager
from tensorrt_engine import build_manifest, manifest_path, write_manifest


def _write_valid_artifacts(tmp_path, *, task="detect", classes=None, class_groups=None, imgsz=960):
    source = tmp_path / "model.pt"
    engine = tmp_path / "model.engine"
    source.write_bytes(b"pytorch-model")
    engine.write_bytes(b"tensorrt-engine")
    write_manifest(
        manifest_path(engine),
        build_manifest(
            source_path=source,
            engine_path=engine,
            imgsz=imgsz,
            precision="fp16",
            task=task,
            classes=classes,
            class_groups=class_groups,
        ),
    )
    return source, engine


def test_coco_runtime_uses_explicit_valid_engine(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path)
    monkeypatch.setenv("SAFETYLENS_COCO_TENSORRT_ENGINE", str(engine))

    runtime_path, backend, fixed_imgsz, fixed_classes, fixed_class_groups, error = model_manager._configured_runtime_path(
        "coco_primary",
        source,
    )

    assert runtime_path == engine
    assert backend == "tensorrt"
    assert fixed_imgsz == 960
    assert fixed_classes == []
    assert fixed_class_groups == []
    assert error is None


def test_coco_runtime_rejects_engine_for_different_source(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path)
    source.write_bytes(b"new-model-release")
    monkeypatch.setenv("SAFETYLENS_COCO_TENSORRT_ENGINE", str(engine))

    runtime_path, backend, fixed_imgsz, fixed_classes, fixed_class_groups, error = model_manager._configured_runtime_path(
        "coco_primary",
        source,
    )

    assert runtime_path == source
    assert backend == "pytorch"
    assert fixed_imgsz is None
    assert fixed_classes == []
    assert fixed_class_groups == []
    assert "source-model hash" in error


def test_ppe_runtime_uses_prompt_bound_engine(tmp_path, monkeypatch):
    classes = ["hard hat", "safety helmet"]
    class_groups = ["helmet_required", "helmet_required"]
    source, engine = _write_valid_artifacts(
        tmp_path,
        task="segment",
        classes=classes,
        class_groups=class_groups,
    )
    monkeypatch.setenv("SAFETYLENS_PPE_TENSORRT_ENGINE", str(engine))

    runtime_path, backend, fixed_imgsz, fixed_classes, fixed_class_groups, error = model_manager._configured_runtime_path(
        "ppe_specialist",
        source,
    )

    assert runtime_path == engine
    assert backend == "tensorrt"
    assert fixed_imgsz == 960
    assert fixed_classes == classes
    assert fixed_class_groups == class_groups
    assert error is None


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


def test_low_resolution_coco_engine_is_selected_only_for_frames_that_fit(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path, imgsz=512)
    calls = []

    class FakeHandle:
        def predict(self, frame, **kwargs):
            calls.append((frame.shape, kwargs))
            return ["low-resolution-result"]

    monkeypatch.setenv("SAFETYLENS_COCO_LOW_RES_TENSORRT_ENGINE", str(engine))
    monkeypatch.setattr(model_manager, "_COCO_LOW_RES_RUNTIME", model_manager._new_model_runtime())
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path, task=None: FakeHandle()))

    skipped, result = model_manager._predict_with_low_res_coco_runtime(
        np.zeros((720, 960, 3), dtype=np.uint8),
        source_path=source,
        conf=0.4,
        device="cuda",
        imgsz=960,
    )
    used, result = model_manager._predict_with_low_res_coco_runtime(
        np.zeros((288, 352, 3), dtype=np.uint8),
        source_path=source,
        conf=0.4,
        device="cuda",
        imgsz=960,
    )

    assert skipped is False
    assert used is True
    assert result == ["low-resolution-result"]
    assert [call[1]["imgsz"] for call in calls] == [512, 512]


def test_coco_model_status_exposes_low_resolution_runtime(monkeypatch):
    runtime = model_manager._new_model_runtime()
    runtime.update(
        runtime_backend="tensorrt",
        runtime_path="/models/yolo26s-512.engine",
        runtime_fallback_error=None,
        fixed_imgsz=512,
        warmed=True,
    )
    monkeypatch.setattr(model_manager, "_COCO_LOW_RES_RUNTIME", runtime)

    status = model_manager._serialize_model_state("coco_primary")

    assert status["runtime_low_res_backend"] == "tensorrt"
    assert status["runtime_low_res_path"] == "/models/yolo26s-512.engine"
    assert status["runtime_low_res_fallback_error"] is None
    assert status["runtime_low_res_fixed_imgsz"] == 512
    assert status["runtime_low_res_warmed"] is True


def test_low_resolution_coco_failure_falls_through_to_primary_runtime(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path, imgsz=512)

    class FailingHandle:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("candidate execution failed")

    class PrimaryHandle:
        def predict(self, _frame, **_kwargs):
            return ["primary-result"]

    primary_runtime = model_manager._new_model_runtime()
    primary_runtime.update(handle=PrimaryHandle(), runtime_backend="tensorrt", fixed_imgsz=960, warmed=True)
    monkeypatch.setenv("SAFETYLENS_COCO_LOW_RES_TENSORRT_ENGINE", str(engine))
    monkeypatch.setattr(model_manager, "_COCO_LOW_RES_RUNTIME", model_manager._new_model_runtime())
    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "coco_primary", primary_runtime)
    monkeypatch.setitem(model_manager._MODEL_STATES["coco_primary"], "active_path", str(source))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path, task=None: FailingHandle()))

    result = model_manager.predict(
        "coco_primary",
        np.zeros((288, 352, 3), dtype=np.uint8),
        conf=0.4,
        device="cuda",
        imgsz=960,
    )

    assert result == ["primary-result"]
    assert model_manager._COCO_LOW_RES_RUNTIME["runtime_backend"] == "tensorrt_failed"


def test_low_resolution_coco_engine_is_warmed_before_camera_inference(tmp_path, monkeypatch):
    source, engine = _write_valid_artifacts(tmp_path, imgsz=512)
    calls = []

    class FakeHandle:
        def predict(self, frame, **kwargs):
            calls.append((frame.shape, kwargs))
            return ["result"]

    monkeypatch.setenv("SAFETYLENS_COCO_LOW_RES_TENSORRT_ENGINE", str(engine))
    monkeypatch.setattr(model_manager, "_COCO_LOW_RES_RUNTIME", model_manager._new_model_runtime())
    monkeypatch.setitem(model_manager._MODEL_STATES["coco_primary"], "active_path", str(source))
    monkeypatch.setitem(model_manager._MODEL_STATES["coco_primary"], "status", "ready")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path, task=None: FakeHandle()))
    monkeypatch.setattr(config_manager, "get_config", lambda: {"global": {"device": "cuda"}})

    assert model_manager._warm_configured_low_res_coco_runtime() is True
    assert [call[1]["imgsz"] for call in calls] == [512, 512]

    used, result = model_manager._predict_with_low_res_coco_runtime(
        np.zeros((288, 352, 3), dtype=np.uint8),
        source_path=source,
        conf=0.4,
        device="cuda",
        imgsz=960,
    )

    assert used is True
    assert result == ["result"]
    assert [call[1]["imgsz"] for call in calls] == [512, 512, 512]


def test_fixed_tensorrt_runtime_is_warmed_before_reporting_ready(monkeypatch):
    calls = []

    class FakeHandle:
        def predict(self, frame, **kwargs):
            calls.append((frame.shape, kwargs))
            return ["result"]

    runtime = model_manager._new_model_runtime()
    runtime.update(
        handle=FakeHandle(),
        runtime_backend="tensorrt",
        fixed_imgsz=960,
        warmed=False,
    )
    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "coco_primary", runtime)
    monkeypatch.setattr(config_manager, "get_config", lambda: {"global": {"device": "cuda"}})

    assert model_manager._warm_configured_fixed_runtime("coco_primary") is True
    assert runtime["warmed"] is True
    assert [call[1]["imgsz"] for call in calls] == [960, 960]

    model_manager._predict_with_runtime(
        "coco_primary",
        runtime,
        np.zeros((32, 32, 3), dtype=np.uint8),
        conf=0.4,
        device="cuda",
        imgsz=960,
    )

    assert [call[1]["imgsz"] for call in calls] == [960, 960, 960]


def test_fixed_runtime_warm_failure_clears_prompt_state_for_fallback(monkeypatch):
    class FailingHandle:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("warm failed")

    runtime = model_manager._new_model_runtime()
    runtime.update(
        handle=FailingHandle(),
        runtime_backend="tensorrt",
        fallback_path="/models/ppe.pt",
        fixed_imgsz=960,
        fixed_classes=["hard hat"],
        fixed_class_groups=["helmet_required"],
        current_classes=["hard hat"],
        class_embeddings={("hard hat",): object()},
    )

    def activate(active_runtime, error):
        assert str(error) == "warm failed"
        active_runtime.update(
            handle=object(),
            runtime_backend="pytorch_fallback",
            fallback_path=None,
            fixed_imgsz=None,
            fixed_classes=[],
            fixed_class_groups=[],
            current_classes=[],
            class_embeddings={},
            warmed=False,
        )

    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "ppe_specialist", runtime)
    monkeypatch.setattr(model_manager, "_activate_pytorch_fallback", activate)
    monkeypatch.setattr(config_manager, "get_config", lambda: {"global": {"device": "cuda"}})

    assert model_manager._warm_configured_fixed_runtime("ppe_specialist") is False
    assert runtime["runtime_backend"] == "pytorch_fallback"
    assert runtime["current_classes"] == []
    assert runtime["class_embeddings"] == {}


def test_ppe_engine_classes_are_checked_after_warm_without_a_second_load(monkeypatch):
    class MismatchedHandle:
        names = {0: "safety vest"}

        def predict(self, *_args, **_kwargs):
            return ["result"]

    runtime = model_manager._new_model_runtime()
    runtime.update(
        handle=MismatchedHandle(),
        runtime_backend="tensorrt",
        fallback_path="/models/ppe.pt",
        fixed_imgsz=960,
        fixed_classes=["hard hat"],
        fixed_class_groups=["helmet_required"],
        current_classes=["hard hat"],
    )
    fallback_errors = []

    def activate(active_runtime, error):
        fallback_errors.append(str(error))
        active_runtime.update(
            handle=object(),
            runtime_backend="pytorch_fallback",
            fallback_path=None,
            fixed_imgsz=None,
            fixed_classes=[],
            fixed_class_groups=[],
            current_classes=[],
            class_embeddings={},
            warmed=False,
        )

    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "ppe_specialist", runtime)
    monkeypatch.setattr(model_manager, "_activate_pytorch_fallback", activate)
    monkeypatch.setattr(config_manager, "get_config", lambda: {"global": {"device": "cuda"}})

    assert model_manager._warm_configured_fixed_runtime("ppe_specialist") is False
    assert fallback_errors == ["TensorRT engine classes do not match its manifest"]
    assert runtime["runtime_backend"] == "pytorch_fallback"


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
    assert runtime["fixed_classes"] == []
    assert runtime["fixed_class_groups"] == []


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
    assert runtime["fixed_class_groups"] == []
    assert runtime["current_classes"] == []
    assert runtime["class_embeddings"] == {}
    assert runtime["warmed"] is False


def test_fixed_prompt_drift_falls_back_before_engine_execution(monkeypatch):
    class EngineHandle:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("fixed engine executed with different prompts")

    class FallbackHandle:
        def __init__(self):
            self.calls = []

        def predict(self, _frame, **kwargs):
            self.calls.append(kwargs)
            return ["pytorch-result"]

    runtime = model_manager._new_model_runtime()
    runtime.update(
        handle=EngineHandle(),
        runtime_backend="tensorrt",
        fallback_path="/models/ppe.pt",
        fixed_imgsz=960,
        fixed_classes=["hard hat"],
        fixed_class_groups=["helmet_required"],
        current_classes=["hard hat"],
        warmed=True,
    )
    fallback = FallbackHandle()

    def activate(active_runtime, error):
        assert "do not match" in str(error)
        active_runtime.update(
            handle=fallback,
            runtime_backend="pytorch_fallback",
            fallback_path=None,
            fixed_imgsz=None,
            fixed_classes=[],
            fixed_class_groups=[],
            current_classes=["safety helmet"],
            warmed=True,
        )

    monkeypatch.setitem(model_manager._MODEL_RUNTIMES, "ppe_specialist", runtime)
    monkeypatch.setattr(model_manager, "_activate_pytorch_fallback", activate)

    result = model_manager.predict(
        "ppe_specialist",
        np.zeros((32, 32, 3), dtype=np.uint8),
        conf=0.4,
        device="cuda",
        imgsz=640,
        classes=["safety helmet"],
    )

    assert result == ["pytorch-result"]
    assert fallback.calls[0]["imgsz"] == 640


def test_same_capability_prompt_synonyms_are_deduplicated_at_high_iou():
    records = [
        {"class_id": 0, "confidence": 0.90, "bbox": [10, 10, 50, 50]},
        {"class_id": 1, "confidence": 0.70, "bbox": [10, 10, 50, 50]},
        {"class_id": 1, "confidence": 0.60, "bbox": [60, 10, 100, 50]},
    ]

    deduplicated = model_manager._deduplicate_prompt_synonyms(
        records,
        ["motorcycle helmet", "rider helmet"],
        class_groups=["rider_helmet_required", "rider_helmet_required"],
    )

    assert deduplicated == [records[0], records[2]]
