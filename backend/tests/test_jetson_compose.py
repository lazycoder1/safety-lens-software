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
