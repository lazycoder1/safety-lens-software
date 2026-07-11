import json
import sys
from types import SimpleNamespace

import pytest

import scripts.export_tensorrt_engine as export_module
from scripts.export_tensorrt_engine import export_engine


def test_fixed_prompt_export_freezes_classes_and_writes_manifest(tmp_path, monkeypatch):
    source = tmp_path / "ppe.pt"
    output = tmp_path / "ppe.engine"
    encoder = tmp_path / "mobileclip2_b.ts"
    source.write_bytes(b"pytorch-model")
    encoder.write_bytes(b"text-encoder")
    created_models = []

    class FakeModel:
        task = "segment"

        def __init__(self, path):
            self.path = path
            self.classes = None

        def set_classes(self, classes):
            self.classes = list(classes)

        def export(self, **kwargs):
            exported = self.path.with_suffix(".engine")
            exported.write_bytes(b"tensorrt-engine")
            assert kwargs["dynamic"] is False
            assert kwargs["half"] is True
            return str(exported)

    def fake_yolo(path):
        model = FakeModel(type(source)(path))
        created_models.append(model)
        return model

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="test-version"))
    monkeypatch.setattr(export_module, "_require_fixed_prompt_dependencies", lambda: None)

    report = export_engine(
        source_path=source,
        output_path=output,
        imgsz=960,
        workspace=2.0,
        device=0,
        force=False,
        classes=["hard hat", "safety helmet"],
        class_groups=["helmet_required", "helmet_required"],
        text_encoder_path=encoder,
    )

    assert created_models[0].classes == ["hard hat", "safety helmet"]
    assert output.read_bytes() == b"tensorrt-engine"
    manifest = json.loads(output.with_suffix(".engine.json").read_text())
    assert manifest["task"] == "segment"
    assert manifest["classes"] == ["hard hat", "safety helmet"]
    assert manifest["classGroups"] == ["helmet_required", "helmet_required"]
    assert report["engine"] == str(output)


def test_fixed_prompt_export_fails_before_model_load_when_clip_is_missing(tmp_path, monkeypatch):
    source = tmp_path / "ppe.pt"
    output = tmp_path / "ppe.engine"
    encoder = tmp_path / "mobileclip2_b.ts"
    source.write_bytes(b"pytorch-model")
    encoder.write_bytes(b"text-encoder")
    model_loaded = False

    def fake_yolo(_path):
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model loading must not start without export dependencies")

    def missing_dependency():
        raise RuntimeError("pinned CLIP missing")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.setattr(export_module, "_require_fixed_prompt_dependencies", missing_dependency)

    with pytest.raises(RuntimeError, match="pinned CLIP missing"):
        export_engine(
            source_path=source,
            output_path=output,
            imgsz=512,
            workspace=2.0,
            device=0,
            force=False,
            classes=["helmet"],
            class_groups=["rider_helmet_required"],
            text_encoder_path=encoder,
        )

    assert model_loaded is False
    assert output.exists() is False
