import json

from tensorrt_engine import build_manifest, manifest_path, validate_engine, write_manifest


def _write_valid_artifacts(tmp_path):
    source = tmp_path / "model.pt"
    engine = tmp_path / "model.engine"
    source.write_bytes(b"pytorch-model")
    engine.write_bytes(b"tensorrt-engine")
    payload = build_manifest(
        source_path=source,
        engine_path=engine,
        imgsz=960,
        precision="fp16",
        task="detect",
    )
    write_manifest(manifest_path(engine), payload)
    return source, engine, payload


def test_validate_engine_accepts_matching_artifacts(tmp_path):
    source, engine, expected = _write_valid_artifacts(tmp_path)

    payload, error = validate_engine(
        source_path=source,
        engine_path=engine,
        expected_task="detect",
    )

    assert error is None
    assert payload == expected
    assert payload["imgsz"] == 960


def test_validate_engine_rejects_missing_manifest(tmp_path):
    source = tmp_path / "model.pt"
    engine = tmp_path / "model.engine"
    source.write_bytes(b"pytorch-model")
    engine.write_bytes(b"tensorrt-engine")

    payload, error = validate_engine(
        source_path=source,
        engine_path=engine,
        expected_task="detect",
    )

    assert payload is None
    assert "manifest does not exist" in error


def test_validate_engine_rejects_changed_source_model(tmp_path):
    source, engine, _payload = _write_valid_artifacts(tmp_path)
    source.write_bytes(b"different-pytorch-model")

    payload, error = validate_engine(
        source_path=source,
        engine_path=engine,
        expected_task="detect",
    )

    assert payload is None
    assert error == "TensorRT source-model hash does not match the active model"


def test_validate_engine_rejects_changed_engine(tmp_path):
    source, engine, _payload = _write_valid_artifacts(tmp_path)
    engine.write_bytes(b"different-tensorrt-engine")

    payload, error = validate_engine(
        source_path=source,
        engine_path=engine,
        expected_task="detect",
    )

    assert payload is None
    assert error == "TensorRT engine hash does not match its manifest"


def test_validate_engine_rejects_mislabeled_manifest(tmp_path):
    source, engine, payload = _write_valid_artifacts(tmp_path)
    payload["engineFile"] = "another.engine"
    manifest_path(engine).write_text(json.dumps(payload), encoding="utf-8")

    validated, error = validate_engine(
        source_path=source,
        engine_path=engine,
        expected_task="detect",
    )

    assert validated is None
    assert error == "TensorRT manifest engine filename does not match the configured engine"
