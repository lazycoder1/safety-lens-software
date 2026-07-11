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
    assert set(cfg["cameras"]) == {"cam2"}
    assert cfg["cameras"]["cam2"]["name"] == "Warehouse Aisle"


def test_load_config_reads_existing_file():
    custom = {"global": {"target_fps": 10}, "vlm": {}, "cameras": {}}
    _test_config.write_text(json.dumps(custom))
    cfg = config_manager.load_config()
    assert cfg["global"]["target_fps"] == 10
    assert cfg["cameras"] == {}


def test_load_config_normalizes_legacy_camera_rule_fields():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_food": {
                "name": "Food Factory",
                "demo": "yoloe",
                "rules": ["Hairnet Detection", "Gloves Detection"],
                "yoloe_classes": ["person", "hairnet", "gloves", "face mask"],
                "alert_classes": ["mobile_phone"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_food"]

    assert "ppe_hairnet" in camera["safety_rule_ids"]
    assert "ppe_gloves" in camera["safety_rule_ids"]
    assert "ppe_face_mask" in camera["safety_rule_ids"]
    assert "alert_mobile_phone" in camera["safety_rule_ids"]
    assert "mobile_phone" in camera["alert_classes"]
    assert "ppe_hairnet" in camera["ppe_rule_ids"]
    assert "person" in camera["yoloe_classes"]
    assert "hairnet_required" in camera["capabilities"]
    assert camera["profile"] == "work_zone_ppe"
    assert camera["execution_plan"]["run_ppe_specialist"] is True


def test_load_config_maps_face_mask_ppe_rule_to_person_plus_ppe_plan():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_mask": {
                "name": "Hospital Mask Corridor",
                "demo": "yolo+yoloe",
                "safety_rule_ids": ["ppe_face_mask"],
                "ppe_rule_ids": ["ppe_face_mask"],
                "capabilities": ["person_presence", "face_mask_required"],
                "yoloe_classes": ["person", "face mask"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_mask"]

    assert camera["capabilities"] == ["face_mask_required"]
    assert camera["execution_plan"]["run_coco_primary"] is True
    assert camera["execution_plan"]["run_ppe_specialist"] is True
    assert camera["execution_plan"]["run_yoloe_long_tail"] is False
    assert camera["execution_plan"]["ppe_prompt_terms"] == [
        "face mask",
        "surgical mask",
        "medical mask",
        "mask",
        "respirator",
    ]
    assert "ppe_face_mask" in camera["ppe_rule_ids"]
    assert "surgical mask" in camera["yoloe_classes"]
    assert "medical mask" in camera["yoloe_classes"]
    assert "respirator" in camera["yoloe_classes"]


def test_load_config_maps_legacy_facemask_rule_id_to_face_mask_capability():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_mask_legacy": {
                "name": "Legacy Mask Camera",
                "demo": "yolo+yoloe",
                "safety_rule_ids": ["ppe_facemask"],
                "ppe_rule_ids": ["ppe_facemask"],
                "yoloe_classes": ["person", "face mask"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_mask_legacy"]

    assert camera["capabilities"] == ["face_mask_required"]
    assert camera["execution_plan"]["run_coco_primary"] is True
    assert camera["execution_plan"]["run_ppe_specialist"] is True
    assert "medical mask" in camera["execution_plan"]["ppe_prompt_terms"]


def test_load_config_rebuilds_capabilities_from_safety_rule_ids_when_stale():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_stale": {
                "name": "TMEIC PE Stores",
                "demo": "yolo+yoloe",
                "rules": ["Person Detection", "Mobile Phone Usage"],
                "yoloe_classes": ["person", "safety vest"],
                "safety_rule_ids": ["ppe_vest", "ppe_helmet", "alert_mobile_phone"],
                "ppe_rule_ids": ["ppe_vest"],
                "capabilities": ["person_presence", "mobile_phone", "vest_required"],
                "execution_plan": {
                    "capabilities": ["person_presence", "mobile_phone", "vest_required"],
                    "ppe_prompt_terms": ["safety vest"],
                    "required_model_keys": ["coco_primary", "ppe_specialist"],
                    "run_coco_primary": True,
                    "run_ppe_specialist": True,
                    "run_yoloe_long_tail": False,
                },
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_stale"]

    assert "ppe_helmet" in camera["ppe_rule_ids"]
    assert "helmet_required" in camera["capabilities"]
    assert "hard hat" in camera["yoloe_classes"]
    assert "safety helmet" in camera["yoloe_classes"]
    assert "hard hat" in camera["execution_plan"]["ppe_prompt_terms"]
    assert camera["execution_plan"]["run_ppe_specialist"] is True


def test_load_config_uses_yaml_ppe_rule_classes_as_prompt_terms():
    custom = {
        "global": {},
        "vlm": {},
        "safety_rules": [
            {
                "id": "ppe_vest",
                "name": "Safety vest",
                "type": "ppe",
                "classes": ["safety vest", "reflective vest", "construction vest"],
                "model": "yoloe",
                "severity": "P2",
                "enabled": True,
            }
        ],
        "cameras": {
            "cam_vest": {
                "name": "Construction Vest Camera",
                "demo": "yolo+yoloe",
                "safety_rule_ids": ["ppe_vest"],
                "capabilities": ["vest_required"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_vest"]

    assert camera["execution_plan"]["run_ppe_specialist"] is True
    assert "reflective vest" in camera["execution_plan"]["ppe_prompt_terms"]
    assert "construction vest" in camera["execution_plan"]["ppe_prompt_terms"]
    assert camera["execution_plan"]["yoloe_prompt_terms"] == []
    assert camera["custom_long_tail_terms"] == []
    assert "reflective vest" in camera["yoloe_classes"]


def test_load_config_maps_harness_ppe_rule_to_person_plus_ppe_plan():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_height": {
                "name": "Work At Height",
                "demo": "yolo+yoloe",
                "safety_rule_ids": ["ppe_harness"],
                "ppe_rule_ids": ["ppe_harness"],
                "capabilities": ["person_presence", "harness_required"],
                "yoloe_classes": ["person", "safety harness"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_height"]

    assert camera["capabilities"] == ["harness_required"]
    assert camera["execution_plan"]["run_coco_primary"] is True
    assert camera["execution_plan"]["run_ppe_specialist"] is True
    assert camera["execution_plan"]["run_yoloe_long_tail"] is False
    assert "safety harness" in camera["execution_plan"]["ppe_prompt_terms"]
    assert "safety lanyard" in camera["execution_plan"]["ppe_prompt_terms"]
    assert "body harness" in camera["yoloe_classes"]
    assert "fall protection lanyard" in camera["yoloe_classes"]


def test_load_config_maps_apron_ppe_rule_to_expanded_apron_prompts():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_apron": {
                "name": "Cafe PPE",
                "demo": "yolo+yoloe",
                "safety_rule_ids": ["ppe_apron"],
                "ppe_rule_ids": ["ppe_apron"],
                "capabilities": ["person_presence", "apron_required"],
                "yoloe_classes": ["person", "apron"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_apron"]

    assert camera["capabilities"] == ["apron_required"]
    assert camera["execution_plan"]["run_coco_primary"] is True
    assert camera["execution_plan"]["run_ppe_specialist"] is True
    assert camera["execution_plan"]["run_yoloe_long_tail"] is False
    assert "denim apron" in camera["execution_plan"]["ppe_prompt_terms"]
    assert "work apron" in camera["yoloe_classes"]


def test_load_config_allows_yaml_closed_set_candidate_override_for_apron():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_apron_candidate": {
                "name": "Cafe PPE Candidate",
                "demo": "yolo",
                "safety_rule_ids": ["ppe_apron"],
                "ppe_rule_ids": ["ppe_apron"],
                "capabilities": ["apron_required"],
                "capability_model_overrides": {
                    "apron_required": "ppe_closed_set_candidate",
                },
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_apron_candidate"]

    assert camera["capabilities"] == ["apron_required"]
    assert camera["execution_plan"]["required_model_keys"] == ["ppe_closed_set_candidate"]
    assert camera["execution_plan"]["capability_model_overrides"] == {
        "apron_required": "ppe_closed_set_candidate",
    }
    assert camera["execution_plan"]["run_ppe_specialist"] is False
    assert camera["execution_plan"]["run_ppe_closed_set_candidate"] is True
    assert camera["execution_plan"]["ppe_prompt_terms"] == []
    assert camera["yoloe_classes"] == []


def test_load_config_maps_boots_ppe_rule_to_expanded_boot_prompts():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_sanitation": {
                "name": "Sanitation Bridge",
                "demo": "yolo+yoloe",
                "safety_rule_ids": ["ppe_boots"],
                "ppe_rule_ids": ["ppe_boots"],
                "capabilities": ["person_presence", "boots_required"],
                "yoloe_classes": ["person", "safety boots"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_sanitation"]

    assert camera["capabilities"] == ["boots_required"]
    assert camera["execution_plan"]["run_coco_primary"] is True
    assert camera["execution_plan"]["run_ppe_specialist"] is True
    assert camera["execution_plan"]["run_yoloe_long_tail"] is False
    assert "rubber boots" in camera["execution_plan"]["ppe_prompt_terms"]
    assert "protective boots" in camera["yoloe_classes"]


def test_load_config_maps_face_shield_ppe_rule_to_person_plus_ppe_plan():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_face_shield": {
                "name": "Hospital PPE Corridor",
                "demo": "yolo+yoloe",
                "safety_rule_ids": ["ppe_face_shield"],
                "ppe_rule_ids": ["ppe_face_shield"],
                "capabilities": ["person_presence", "face_shield_required"],
                "yoloe_classes": ["person", "face shield"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_face_shield"]

    assert camera["capabilities"] == ["face_shield_required"]
    assert camera["execution_plan"]["run_coco_primary"] is True
    assert camera["execution_plan"]["run_ppe_specialist"] is True
    assert camera["execution_plan"]["run_yoloe_long_tail"] is False
    assert camera["execution_plan"]["ppe_prompt_terms"] == [
        "face shield",
        "protective face shield",
        "clear face shield",
        "visor",
        "protective visor",
    ]
    assert "ppe_face_shield" in camera["ppe_rule_ids"]
    assert "protective face shield" in camera["yoloe_classes"]
    assert "protective visor" in camera["yoloe_classes"]


def test_load_config_ignores_stale_rule_labels_when_rule_ids_exist():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_stale": {
                "name": "Drifted Camera",
                "rules": ["Mobile Phone Usage", "Animal Intrusion"],
                "safety_rule_ids": ["ppe_helmet"],
                "ppe_rule_ids": ["ppe_helmet"],
                "capabilities": ["mobile_phone", "animal_presence"],
                "yoloe_classes": ["person", "hard hat", "safety helmet"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_stale"]

    assert "helmet_required" in camera["capabilities"]
    assert "mobile_phone" not in camera["capabilities"]
    assert "animal_presence" not in camera["capabilities"]
    assert camera["execution_plan"]["run_ppe_specialist"] is True


def test_load_config_ignores_stale_legacy_alert_fields_when_rule_ids_exist():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_stale": {
                "name": "Detection Drift Camera",
                "safety_rule_ids": ["ppe_hairnet"],
                "alert_classes": ["mobile_phone", "animal_intrusion"],
                "yoloe_classes": ["person", "hairnet", "cell phone"],
                "capabilities": ["mobile_phone", "animal_presence", "hairnet_required"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_stale"]

    assert camera["safety_rule_ids"] == ["ppe_hairnet"]
    assert camera["alert_classes"] == []
    assert camera["capabilities"] == ["hairnet_required"]
    assert "mobile_phone" not in camera["capabilities"]
    assert "animal_presence" not in camera["capabilities"]


def test_load_config_ignores_stale_ppe_rule_ids_when_rule_ids_exist():
    custom = {
        "global": {},
        "vlm": {},
        "cameras": {
            "cam_stale": {
                "name": "PPE Drift Camera",
                "safety_rule_ids": ["alert_animal"],
                "ppe_rule_ids": ["ppe_helmet", "ppe_hairnet"],
                "capabilities": ["animal_presence", "helmet_required", "hairnet_required"],
                "yoloe_classes": ["person", "hard hat", "hairnet"],
            }
        },
    }
    _test_config.write_text(json.dumps(custom))

    cfg = config_manager.load_config()
    camera = cfg["cameras"]["cam_stale"]

    assert camera["safety_rule_ids"] == ["alert_animal"]
    assert camera["ppe_rule_ids"] == []
    assert camera["capabilities"] == ["animal_presence"]
    assert "helmet_required" not in camera["capabilities"]
    assert "hairnet_required" not in camera["capabilities"]


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
    """Verify no temporary config file remains after save."""
    cfg = {"global": {"target_fps": 8}, "vlm": {}, "cameras": {}}
    config_manager.save_config(cfg)
    tmp = Path(str(_test_config) + ".tmp")
    assert not tmp.exists()
    assert not list(_test_config.parent.glob(f"{_test_config.name}.*.tmp"))


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
    cam_id = next(iter(config_manager.get_config()["cameras"]))
    config_manager.update_config(f"cameras.{cam_id}.fps", 12)
    cfg = config_manager.get_config()
    assert cfg["cameras"][cam_id]["fps"] == 12


def test_update_config_returns_full_config():
    config_manager.load_config()
    result = config_manager.update_config("global.yolo_conf", 0.5)
    assert "global" in result
    assert "vlm" in result
    assert "cameras" in result


# ── DEFAULT_CONFIG structure ─────────────────────────────────────────────────

def test_default_config_camera_structure():
    cfg = config_manager.DEFAULT_CONFIG
    cam2 = cfg["cameras"]["cam2"]
    assert cam2["name"] == "Warehouse Aisle"
    assert cam2["demo"] == "yolo+vlm"
    assert "Gangway Blockage (VLM)" in cam2["rules"]


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
