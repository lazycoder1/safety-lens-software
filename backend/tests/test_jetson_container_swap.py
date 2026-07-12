from scripts import jetson_container_swap


def _container_info():
    return {
        "Id": "a" * 64,
        "Config": {
            "Hostname": "aaaaaaaaaaaa",
            "User": "",
            "Env": ["TOKEN=secret", "MODE=batch4"],
            "Cmd": ["python", "server.py"],
            "Entrypoint": None,
            "WorkingDir": "/app",
            "Labels": {"service": "edge"},
            "StopSignal": "SIGTERM",
            "StopTimeout": None,
        },
        "HostConfig": {
            "AutoRemove": False,
            "Binds": ["/host/config.json:/app/config.json:ro"],
            "CapAdd": None,
            "CapDrop": None,
            "CpuShares": 0,
            "CpusetCpus": "",
            "Devices": [
                {
                    "PathOnHost": "/dev/nvhost-gpu",
                    "PathInContainer": "/dev/nvhost-gpu",
                    "CgroupPermissions": "rwm",
                }
            ],
            "DeviceRequests": None,
            "Dns": [],
            "DnsOptions": [],
            "DnsSearch": [],
            "ExtraHosts": None,
            "GroupAdd": None,
            "IpcMode": "private",
            "Init": None,
            "Links": None,
            "LogConfig": {"Type": "json-file", "Config": {}},
            "Memory": 0,
            "MemorySwap": 0,
            "NanoCpus": 0,
            "NetworkMode": "rakshak-net",
            "PidMode": "",
            "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8000"}]},
            "Privileged": False,
            "ReadonlyRootfs": False,
            "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
            "Runtime": "nvidia",
            "SecurityOpt": None,
            "ShmSize": 67108864,
            "Sysctls": None,
            "Tmpfs": None,
            "Ulimits": None,
            "UTSMode": "",
            "VolumesFrom": None,
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/host/config.json",
                "Destination": "/app/config.json",
                "RW": False,
            }
        ],
        "NetworkSettings": {"Networks": {"rakshak-net": {}}},
    }


def _pairs(args, flag):
    return [args[index + 1] for index, value in enumerate(args[:-1]) if value == flag]


def test_build_create_args_clones_runtime_without_shell_interpolation():
    args = jetson_container_swap.build_create_args(
        _container_info(),
        name="rakshak-edge",
        image="edge:candidate",
        env_overrides={"MODE": "adaptive", "EARLY_FLUSH": "0.006"},
    )

    assert args[:3] == ["create", "--name", "rakshak-edge"]
    assert _pairs(args, "--restart") == ["unless-stopped"]
    assert _pairs(args, "--runtime") == ["nvidia"]
    assert _pairs(args, "--network") == ["rakshak-net"]
    assert _pairs(args, "--publish") == ["8000:8000/tcp"]
    assert _pairs(args, "--device") == ["/dev/nvhost-gpu:/dev/nvhost-gpu:rwm"]
    assert _pairs(args, "--env") == [
        "TOKEN=secret",
        "MODE=adaptive",
        "EARLY_FLUSH=0.006",
    ]
    assert args[-3:] == ["edge:candidate", "python", "server.py"]


def test_build_create_args_refuses_unreconstructable_network_topology():
    info = _container_info()
    info["NetworkSettings"]["Networks"]["other-net"] = {}

    try:
        jetson_container_swap.build_create_args(
            info,
            name="rakshak-edge",
            image="edge:candidate",
            env_overrides={},
        )
    except jetson_container_swap.SwapError as exc:
        assert "multiple networks" in str(exc)
    else:
        raise AssertionError("multiple networks must fail closed")


def test_build_create_args_can_repair_an_incorrect_runtime():
    info = _container_info()
    info["HostConfig"]["Runtime"] = "runc"

    args = jetson_container_swap.build_create_args(
        info,
        name="rakshak-edge",
        image="edge:candidate",
        env_overrides={},
        runtime_override="nvidia",
    )

    assert _pairs(args, "--runtime") == ["nvidia"]


def test_parse_env_file_and_explicit_override(tmp_path):
    env_file = tmp_path / "profile.env"
    env_file.write_text("# profile\nMODE=batch4\nEARLY=0.004\n", encoding="utf-8")

    parsed = jetson_container_swap._parse_env(
        ["EARLY=0.006"],
        env_file,
    )

    assert parsed == {"MODE": "batch4", "EARLY": "0.006"}


def test_missing_container_inspect_is_not_treated_as_existing(monkeypatch):
    monkeypatch.setattr(
        jetson_container_swap, "_docker", lambda *_args, **_kwargs: "[]"
    )

    assert not jetson_container_swap._container_exists("missing")


def test_restore_preserves_displaced_candidate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        jetson_container_swap,
        "_inspect_container",
        lambda name: {"Name": name},
    )
    monkeypatch.setattr(
        jetson_container_swap,
        "_container_exists",
        lambda _name: False,
    )
    monkeypatch.setattr(
        jetson_container_swap,
        "_docker",
        lambda *args, **_kwargs: calls.append(args) or "",
    )
    monkeypatch.setattr(
        jetson_container_swap,
        "_wait_for_health",
        lambda url, timeout, **kwargs: calls.append(("health", url, timeout, kwargs)),
    )

    result = jetson_container_swap.restore(
        active="edge",
        rollback="edge-old",
        displaced="edge-candidate",
        health_url="http://127.0.0.1/health",
        health_timeout=30,
        required_fresh_cameras=(),
        required_camera_backends={},
        dry_run=False,
    )

    assert result["status"] == "restored"
    assert calls == [
        ("stop", "--time", "20", "edge"),
        ("rename", "edge", "edge-candidate"),
        ("rename", "edge-old", "edge"),
        ("start", "edge"),
        (
            "health",
            "http://127.0.0.1/health",
            30,
            {"required_fresh_cameras": (), "required_camera_backends": {}},
        ),
    ]


def test_camera_health_requirements_cover_freshness_and_backend():
    payload = {
        "cameras": [
            {
                "id": "cam2",
                "frameFresh": True,
                "connection": {"captureBackend": "gstreamer_nvdec"},
            }
        ]
    }

    assert (
        jetson_container_swap._camera_requirement_error(
            payload,
            ("cam2",),
            {"cam2": "gstreamer_nvdec"},
        )
        is None
    )
    assert "not fresh" in jetson_container_swap._camera_requirement_error(
        {"cameras": [{**payload["cameras"][0], "frameFresh": False}]},
        ("cam2",),
        {},
    )
    assert "wrong capture backend" in (
        jetson_container_swap._camera_requirement_error(
            payload,
            (),
            {"cam2": "ffmpeg"},
        )
    )


def test_parse_camera_backend_requirements():
    assert jetson_container_swap._parse_camera_backends(
        ["cam2=gstreamer_nvdec", "cam1=ffmpeg"]
    ) == {"cam2": "gstreamer_nvdec", "cam1": "ffmpeg"}
