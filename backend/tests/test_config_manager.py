"""Tests for config_manager module — thread-safe config persistence."""

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

# Patch config path before importing
_tmpdir = tempfile.mkdtemp()
_test_config = Path(_tmpdir) / "test_config.json"

import config_manager

config_manager.CONFIG_PATH = _test_config


@pytest.fixture(autouse=True)
def fresh_config():
    """Reset config state before each test."""
    config_manager._config = None
    config_manager.CONFIG_PATH = _test_config
    if _test_config.exists():
        _test_config.unlink()
    tmp = Path(str(_test_config) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    yield
    if _test_config.exists():
        _test_config.unlink()


# ── load_config ──────────────────────────────────────────────────────────────

def test_load_config_creates_default_when_missing():
    cfg = config_manager.load_config()
    assert _test_config.exists()
    assert "global" in cfg
    assert "vlm" in cfg
    assert "cameras" in cfg


def test_load_config_returns_default_values():
    cfg = config_manager.load_config()
    assert cfg["global"]["target_fps"] == 6
    assert cfg["global"]["yolo_conf"] == 0.35
    assert cfg["vlm"]["model"] == "qwen3-vl:8b"
    assert "cam1" in cfg["cameras"]
    assert "cam2" in cfg["cameras"]
    assert "cam3" in cfg["cameras"]


def test_load_config_reads_existing_file():
    custom = {"global": {"target_fps": 10}, "vlm": {}, "cameras": {}}
    _test_config.write_text(json.dumps(custom))
    cfg = config_manager.load_config()
    assert cfg["global"]["target_fps"] == 10
    assert cfg["cameras"] == {}


# ── get_config ───────────────────────────────────────────────────────────────

def test_get_config_loads_on_first_call():
    cfg = config_manager.get_config()
    assert "global" in cfg


def test_get_config_returns_cached():
    cfg1 = config_manager.get_config()
    cfg2 = config_manager.get_config()
    assert cfg1 is cfg2


# ── save_config ──────────────────────────────────────────────────────────────

def test_save_config_writes_to_disk():
    cfg = {"global": {"target_fps": 12}, "vlm": {}, "cameras": {}}
    config_manager.save_config(cfg)
    assert _test_config.exists()
    loaded = json.loads(_test_config.read_text())
    assert loaded["global"]["target_fps"] == 12


def test_save_config_atomic_write():
    """Verify no .tmp file remains after save."""
    cfg = {"global": {"target_fps": 8}, "vlm": {}, "cameras": {}}
    config_manager.save_config(cfg)
    tmp = Path(str(_test_config) + ".tmp")
    assert not tmp.exists()


def test_save_config_updates_cache():
    cfg = {"global": {"target_fps": 15}, "vlm": {}, "cameras": {}}
    config_manager.save_config(cfg)
    cached = config_manager.get_config()
    assert cached["global"]["target_fps"] == 15


# ── update_config ────────────────────────────────────────────────────────────

def test_update_config_single_key():
    config_manager.load_config()
    result = config_manager.update_config("global.target_fps", 10)
    assert result["global"]["target_fps"] == 10

    # Verify persisted
    loaded = json.loads(_test_config.read_text())
    assert loaded["global"]["target_fps"] == 10


def test_update_config_nested_key():
    config_manager.load_config()
    config_manager.update_config("vlm.model", "qwen3.5:35b")
    cfg = config_manager.get_config()
    assert cfg["vlm"]["model"] == "qwen3.5:35b"


def test_update_config_camera_property():
    config_manager.load_config()
    config_manager.update_config("cameras.cam1.fps", 12)
    cfg = config_manager.get_config()
    assert cfg["cameras"]["cam1"]["fps"] == 12


def test_update_config_returns_full_config():
    config_manager.load_config()
    result = config_manager.update_config("global.yolo_conf", 0.5)
    assert "global" in result
    assert "vlm" in result
    assert "cameras" in result


# ── DEFAULT_CONFIG structure ─────────────────────────────────────────────────

def test_default_config_camera_structure():
    cfg = config_manager.DEFAULT_CONFIG
    cam1 = cfg["cameras"]["cam1"]
    assert cam1["name"] == "Welding Bay"
    assert cam1["demo"] == "yolo"
    assert "Hard Hat Detection" in cam1["rules"]

    cam3 = cfg["cameras"]["cam3"]
    assert cam3["demo"] == "yoloe"
    assert "person" in cam3["yoloe_classes"]


def test_default_config_vlm_keywords():
    cfg = config_manager.DEFAULT_CONFIG
    kw = cfg["vlm"]["violation_keywords"]
    assert "not wearing" in kw
    assert "blocked" in kw


# ── database + telegram sections ─────────────────────────────────────────────

def test_default_config_has_database_section():
    cfg = config_manager.DEFAULT_CONFIG
    assert "database" in cfg
    assert "url" in cfg["database"]
    assert "postgresql" in cfg["database"]["url"]


def test_default_config_has_telegram_section():
    cfg = config_manager.DEFAULT_CONFIG
    assert "telegram" in cfg
    assert "enabled" in cfg["telegram"]
    assert "bot_token" in cfg["telegram"]
    assert "chat_id" in cfg["telegram"]
    assert "severities" in cfg["telegram"]


def test_telegram_section_defaults():
    cfg = config_manager.DEFAULT_CONFIG
    tg = cfg["telegram"]
    assert tg["enabled"] is False
    assert tg["bot_token"] == ""
    assert tg["chat_id"] == ""
    assert tg["severities"] == ["P1", "P2"]


# ── thread safety ────────────────────────────────────────────────────────────

def test_concurrent_updates():
    config_manager.load_config()
    errors = []

    def worker(n):
        try:
            for i in range(20):
                config_manager.update_config("global.target_fps", n * 100 + i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    # Config should be valid JSON
    loaded = json.loads(_test_config.read_text())
    assert isinstance(loaded["global"]["target_fps"], int)
