from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_jetson_edge_exposes_nvdec_runtime_devices_and_plugins():
    compose = yaml.safe_load((ROOT / "docker-compose.jetson.yml").read_text())
    edge = compose["services"]["edge"]

    assert edge["environment"]["SAFETYLENS_RTSP_CAPTURE_BACKEND"] == "nvdec"
    assert set(edge["devices"]) >= {
        "/dev/nvidiactl:/dev/nvidiactl",
        "/dev/nvhost-gpu:/dev/nvhost-gpu",
        "/dev/nvhost-ctrl-gpu:/dev/nvhost-ctrl-gpu",
        "/dev/nvmap:/dev/nvmap",
        "/dev/nvhost-ctrl:/dev/nvhost-ctrl",
        "/dev/nvhost-nvdec:/dev/nvhost-nvdec",
        "/dev/nvhost-ctrl-nvdec:/dev/nvhost-ctrl-nvdec",
        "/dev/nvhost-vic:/dev/nvhost-vic",
    }
    assert set(edge["volumes"]) >= {
        "/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so:ro",
        "/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so:ro",
        "/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro",
    }


def test_jetson_model_server_image_contains_anpr_runtime_module():
    dockerfile = (ROOT / "Dockerfile.model-server-jetson").read_text()

    assert "COPY backend/plate_analyzer.py ./backend/plate_analyzer.py" in dockerfile


def test_jetson_model_server_image_contains_rtdetr_phone_runtime_module():
    dockerfile = (ROOT / "Dockerfile.model-server-jetson").read_text()

    assert (
        "COPY backend/rtdetr_phone_runtime.py ./backend/rtdetr_phone_runtime.py"
        in dockerfile
    )


def test_jetson_edge_uses_measured_inference_admission_defaults():
    compose = yaml.safe_load((ROOT / "docker-compose.jetson.yml").read_text())
    environment = compose["services"]["edge"]["environment"]

    assert environment["SAFETYLENS_REMOTE_INFERENCE_MAX_INFLIGHT"] == (
        "${SAFETYLENS_REMOTE_INFERENCE_MAX_INFLIGHT:-4}"
    )
    assert environment["SAFETYLENS_REMOTE_INFERENCE_ADMISSION_WAIT_SECONDS"] == (
        "${SAFETYLENS_REMOTE_INFERENCE_ADMISSION_WAIT_SECONDS:-0.125}"
    )
    assert environment["SAFETYLENS_REMOTE_PRIMARY_BATCH_WAIT_SECONDS"].endswith(
        ":-0.014}"
    )
    assert environment["SAFETYLENS_REMOTE_SPECIALIST_BATCH_WAIT_SECONDS"].endswith(
        ":-0.014}"
    )
    assert environment["SAFETYLENS_REMOTE_FRAME_BATCH_SIZE"].endswith(":-4}")
    assert environment["SAFETYLENS_REMOTE_BATCH2_EARLY_FLUSH_SECONDS"].endswith(
        ":-0.006}"
    )
    assert environment["SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE"].endswith(":-4}")
    assert environment["SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT"].endswith(
        ":-0.70}"
    )
    assert environment["SAFETYLENS_REMOTE_RTDETR_PHONE_BATCH_WAIT_SECONDS"] == (
        "${SAFETYLENS_REMOTE_RTDETR_PHONE_BATCH_WAIT_SECONDS:-0.014}"
    )


def test_jetson_rtdetr_phone_engines_are_optional_and_identity_pinned():
    compose = yaml.safe_load((ROOT / "docker-compose.jetson.yml").read_text())
    environment = compose["services"]["model-server"]["environment"]

    assert environment["SAFETYLENS_RTDETR_PHONE_BATCH1_TENSORRT_ENGINE"].endswith(
        ":-}"
    )
    assert environment["SAFETYLENS_RTDETR_PHONE_BATCH1_ENGINE_SHA256"].endswith(
        ":-}"
    )
    assert environment["SAFETYLENS_RTDETR_PHONE_BATCH2_TENSORRT_ENGINE"].endswith(
        ":-}"
    )
    assert environment["SAFETYLENS_RTDETR_PHONE_BATCH2_ENGINE_SHA256"].endswith(
        ":-}"
    )


def test_split_services_default_to_same_small_coco_model():
    compose = yaml.safe_load((ROOT / "docker-compose.split.yml").read_text())

    for service_name in ("edge", "model-server"):
        assert compose["services"][service_name]["environment"]["SAFETYLENS_COCO_MODEL"] == (
            "${SAFETYLENS_COCO_MODEL:-yolo26s}"
        )


def test_monolithic_backend_defaults_to_small_coco_model():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    assert compose["services"]["backend"]["environment"]["SAFETYLENS_COCO_MODEL"] == (
        "${SAFETYLENS_COCO_MODEL:-yolo26s}"
    )
