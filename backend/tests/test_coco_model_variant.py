import logging

import model_manager


def test_coco_model_defaults_to_small_when_unconfigured():
    assert model_manager._resolve_coco_model_variant(None) == "yolo26s"
    assert model_manager._resolve_coco_model_variant("") == "yolo26s"


def test_coco_model_preserves_explicit_supported_variant():
    assert model_manager._resolve_coco_model_variant("yolo26n") == "yolo26n"
    assert model_manager._resolve_coco_model_variant(" yolo26s ") == "yolo26s"
    assert model_manager._resolve_coco_model_variant("yolo26m") == "yolo26m"


def test_unknown_coco_model_falls_back_to_small(caplog):
    with caplog.at_level(logging.WARNING, logger="rakshak_lens.models"):
        selected = model_manager._resolve_coco_model_variant("typo")

    assert selected == "yolo26s"
    assert "falling back to small" in caplog.text
