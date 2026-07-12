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
            assert kwargs["batch"] == 1
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


def test_export_fails_before_model_load_when_manifest_helper_is_stale(
    tmp_path, monkeypatch
):
    source = tmp_path / "ppe.pt"
    output = tmp_path / "ppe.engine"
    source.write_bytes(b"pytorch-model")
    model_loaded = False

    def fake_yolo(_path):
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model loading must not start with a stale helper")

    def stale_build_manifest(
        *,
        source_path,
        engine_path,
        imgsz,
        precision,
        task,
        classes,
        class_groups,
        metadata,
    ):
        raise AssertionError("stale helper must be rejected before use")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    monkeypatch.setattr(export_module, "build_manifest", stale_build_manifest)

    with pytest.raises(RuntimeError, match="different revisions.*batch"):
        export_engine(
            source_path=source,
            output_path=output,
            imgsz=640,
            workspace=1.0,
            device=0,
            force=False,
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

    def fake_onnx_export(*, source_path, output_path, imgsz, batch, device, classes):
        assert source_path.name == "yolo26s.pt"
        assert imgsz == 512
        assert batch == 2
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
            "batch": 2,
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
        batch=2,
    )

    with output.open("rb") as handle:
        metadata_size = struct.unpack("<I", handle.read(4))[0]
        metadata = json.loads(handle.read(metadata_size))
        engine = handle.read()
    assert metadata["imgsz"] == [512, 512]
    assert metadata["batch"] == 2
    assert metadata["args"]["half"] is True
    assert engine == b"bare-tensorrt-engine"
    assert report["metadata"]["builder"] == "trtexec_heuristic"
    assert report["batch"] == 2
    assert report["metadata"]["batch"] == 2
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


def test_int8_export_requires_and_records_calibration_data(tmp_path, monkeypatch):
    source = tmp_path / "yolo26s.pt"
    output = tmp_path / "yolo26s-int8.engine"
    calibration = tmp_path / "office-calibration.yaml"
    source.write_bytes(b"pytorch-model")
    calibration.write_text("path: /calibration\ntrain: images\n", encoding="utf-8")

    class FakeModel:
        task = "detect"

        def __init__(self, path):
            self.path = path

        def export(self, **kwargs):
            assert kwargs["half"] is False
            assert kwargs["int8"] is True
            assert kwargs["data"] == str(calibration.resolve())
            exported = self.path.with_suffix(".engine")
            exported.write_bytes(b"int8-engine")
            return str(exported)

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=lambda path: FakeModel(type(source)(path))),
    )
    monkeypatch.setitem(
        sys.modules, "tensorrt", SimpleNamespace(__version__="test-version")
    )

    report = export_engine(
        source_path=source,
        output_path=output,
        imgsz=640,
        workspace=0.5,
        device=0,
        force=False,
        precision="int8",
        calibration_data=calibration,
    )

    assert report["precision"] == "int8"
    assert report["metadata"]["calibrationDataFile"] == calibration.name
    assert len(report["metadata"]["calibrationDataSha256"]) == 64


def test_int8_export_rejects_missing_data_and_low_memory_path(tmp_path):
    source = tmp_path / "yolo26s.pt"
    output = tmp_path / "yolo26s-int8.engine"
    calibration = tmp_path / "office-calibration.yaml"
    source.write_bytes(b"pytorch-model")
    calibration.write_text("train: images\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="calibration dataset YAML"):
        export_engine(
            source_path=source,
            output_path=output,
            imgsz=640,
            workspace=0.5,
            device=0,
            force=False,
            precision="int8",
        )
    with pytest.raises(ValueError, match="low-memory"):
        export_engine(
            source_path=source,
            output_path=output,
            imgsz=640,
            workspace=0.5,
            device=0,
            force=False,
            precision="int8",
            calibration_data=calibration,
            low_memory=True,
        )
