import json
import struct
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
    monkeypatch.setitem(
        sys.modules, "tensorrt", SimpleNamespace(__version__="test-version")
    )
    monkeypatch.setattr(
        export_module, "_require_fixed_prompt_dependencies", lambda: None
    )

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


def test_fixed_prompt_export_fails_before_model_load_when_clip_is_missing(
    tmp_path, monkeypatch
):
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
    monkeypatch.setattr(
        export_module, "_require_fixed_prompt_dependencies", missing_dependency
    )

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


def test_low_memory_export_wraps_trtexec_engine_and_records_builder(
    tmp_path, monkeypatch
):
    source = tmp_path / "yolo26s.pt"
    output = tmp_path / "yolo26s-512.engine"
    trtexec = tmp_path / "trtexec"
    source.write_bytes(b"pytorch-model")
    trtexec.write_bytes(b"binary")

    def fake_onnx_export(*, source_path, output_path, imgsz, device, classes):
        assert source_path.name == "yolo26s.pt"
        assert imgsz == 512
        assert device == 0
        assert classes == []
        output_path.write_bytes(b"onnx-model")

    def fake_run(command, *, check):
        assert check is True
        assert "--workspace=192" in command
        assert "--heuristic" in command
        engine_arg = next(
            value for value in command if value.startswith("--saveEngine=")
        )
        type(output)(engine_arg.partition("=")[2]).write_bytes(b"bare-tensorrt-engine")

    monkeypatch.setitem(
        sys.modules, "tensorrt", SimpleNamespace(__version__="test-version")
    )
    monkeypatch.setattr(export_module, "_run_onnx_export_process", fake_onnx_export)
    monkeypatch.setattr(export_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        export_module,
        "_read_onnx_metadata",
        lambda _path: {
            "batch": 1,
            "imgsz": [512, 512],
            "names": {0: "person", 67: "cell phone"},
            "stride": 32,
            "task": "detect",
            "args": {"half": False},
        },
    )

    report = export_engine(
        source_path=source,
        output_path=output,
        imgsz=512,
        workspace=2.0,
        device=0,
        force=False,
        low_memory=True,
        low_memory_workspace_mib=192,
        trtexec_path=trtexec,
    )

    with output.open("rb") as handle:
        metadata_size = struct.unpack("<I", handle.read(4))[0]
        metadata = json.loads(handle.read(metadata_size))
        engine = handle.read()
    assert metadata["imgsz"] == [512, 512]
    assert metadata["args"]["half"] is True
    assert engine == b"bare-tensorrt-engine"
    assert report["metadata"]["builder"] == "trtexec_heuristic"
    assert report["metadata"]["workspaceMiB"] == 192


def test_low_memory_export_requires_trtexec_before_model_load(tmp_path, monkeypatch):
    source = tmp_path / "yolo26s.pt"
    output = tmp_path / "yolo26s-512.engine"
    source.write_bytes(b"pytorch-model")
    model_loaded = False

    def fake_yolo(_path):
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model loading must not start without trtexec")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.setitem(
        sys.modules, "tensorrt", SimpleNamespace(__version__="test-version")
    )

    with pytest.raises(FileNotFoundError, match="TensorRT builder does not exist"):
        export_engine(
            source_path=source,
            output_path=output,
            imgsz=512,
            workspace=2.0,
            device=0,
            force=False,
            low_memory=True,
            trtexec_path=tmp_path / "missing-trtexec",
        )

    assert model_loaded is False
    assert output.exists() is False
