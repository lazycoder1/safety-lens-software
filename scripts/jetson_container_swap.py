#!/usr/bin/env python3
"""Promote a Jetson container image while preserving a verified rollback."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


class SwapError(RuntimeError):
    """A safe container transition could not be completed."""


def _docker(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise SwapError(message or f"docker {args[0]} failed")
    return completed.stdout.strip()


def _inspect_container(name: str) -> dict[str, Any]:
    try:
        payload = json.loads(_docker("inspect", name))
    except (json.JSONDecodeError, SwapError) as exc:
        raise SwapError(f"container {name!r} is not inspectable") from exc
    if len(payload) != 1:
        raise SwapError(f"container {name!r} inspection was ambiguous")
    return payload[0]


def _container_exists(name: str) -> bool:
    output = _docker("container", "inspect", name, check=False)
    try:
        return bool(json.loads(output))
    except json.JSONDecodeError:
        return False


def _image_id(image: str) -> str:
    output = _docker("image", "inspect", image, "--format", "{{.Id}}")
    if not output:
        raise SwapError(f"image {image!r} is not installed")
    return output


def _parse_env(values: list[str], env_file: Path | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if env_file is not None:
        for line_number, raw_line in enumerate(
            env_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise SwapError(
                    f"invalid environment assignment at {env_file}:{line_number}"
                )
            key, value = line.split("=", 1)
            overrides[key.strip()] = value
    for value in values:
        if "=" not in value:
            raise SwapError(f"invalid environment assignment: {value!r}")
        key, item = value.split("=", 1)
        overrides[key.strip()] = item
    if any(not key for key in overrides):
        raise SwapError("environment variable names cannot be empty")
    return overrides


def _parse_camera_backends(values: list[str]) -> dict[str, str]:
    backends: dict[str, str] = {}
    for value in values:
        camera_id, separator, backend = value.partition("=")
        if not separator or not camera_id.strip() or not backend.strip():
            raise SwapError(f"invalid camera backend requirement: {value!r}")
        backends[camera_id.strip()] = backend.strip()
    return backends


def _merged_environment(existing: list[str], overrides: dict[str, str]) -> list[str]:
    order: list[str] = []
    merged: dict[str, str] = {}
    for assignment in existing:
        key, separator, value = assignment.partition("=")
        if not separator:
            continue
        if key not in merged:
            order.append(key)
        merged[key] = value
    for key, value in overrides.items():
        if key not in merged:
            order.append(key)
        merged[key] = value
    return [f"{key}={merged[key]}" for key in order]


def _append_pair(args: list[str], flag: str, value: Any) -> None:
    if value not in (None, "", [], {}):
        args.extend((flag, str(value)))


def _volume_arguments(info: dict[str, Any]) -> list[str]:
    host_config = info["HostConfig"]
    volumes = list(host_config.get("Binds") or [])
    destinations = {
        binding.split(":", 2)[1] for binding in volumes if binding.count(":") >= 1
    }
    for mount in info.get("Mounts") or []:
        destination = mount.get("Destination")
        if (
            not destination
            or destination in destinations
            or mount.get("Type") == "tmpfs"
        ):
            continue
        source = mount.get("Name") or mount.get("Source")
        if not source:
            raise SwapError(f"cannot clone anonymous mount at {destination}")
        suffix = "" if mount.get("RW", True) else ":ro"
        volumes.append(f"{source}:{destination}{suffix}")
    return volumes


def build_create_args(
    info: dict[str, Any],
    *,
    name: str,
    image: str,
    env_overrides: dict[str, str],
    runtime_override: str | None = None,
) -> list[str]:
    """Build a shell-safe docker-create argv from an inspected container."""
    config = info["Config"]
    host = info["HostConfig"]
    networks = info.get("NetworkSettings", {}).get("Networks") or {}
    if len(networks) > 1:
        raise SwapError("containers attached to multiple networks are not supported")
    if host.get("AutoRemove"):
        raise SwapError("auto-remove containers cannot retain a rollback")
    if host.get("Links") or host.get("VolumesFrom"):
        raise SwapError("legacy links and volumes-from are not supported")

    args = ["create", "--name", name]
    runtime = runtime_override or host.get("Runtime")
    if runtime:
        args.extend(("--runtime", str(runtime)))
    restart = host.get("RestartPolicy") or {}
    restart_name = restart.get("Name") or "no"
    if restart_name == "on-failure" and restart.get("MaximumRetryCount"):
        restart_name += f":{restart['MaximumRetryCount']}"
    args.extend(("--restart", restart_name))

    network_mode = host.get("NetworkMode")
    if network_mode and network_mode not in {"default", "bridge"}:
        if str(network_mode).startswith("container:"):
            raise SwapError("container network namespaces are not supported")
        args.extend(("--network", str(network_mode)))

    old_id = str(info.get("Id") or "")
    hostname = config.get("Hostname")
    if hostname and not old_id.startswith(str(hostname)):
        args.extend(("--hostname", str(hostname)))
    _append_pair(args, "--user", config.get("User"))
    _append_pair(args, "--workdir", config.get("WorkingDir"))
    _append_pair(args, "--stop-signal", config.get("StopSignal"))
    _append_pair(args, "--stop-timeout", config.get("StopTimeout"))

    for label, value in sorted((config.get("Labels") or {}).items()):
        args.extend(("--label", f"{label}={value}"))
    for assignment in _merged_environment(config.get("Env") or [], env_overrides):
        args.extend(("--env", assignment))
    for volume in _volume_arguments(info):
        args.extend(("--volume", volume))
    for destination, options in sorted((host.get("Tmpfs") or {}).items()):
        value = destination if not options else f"{destination}:{options}"
        args.extend(("--tmpfs", value))
    for device in host.get("Devices") or []:
        args.extend(
            (
                "--device",
                ":".join(
                    (
                        device["PathOnHost"],
                        device["PathInContainer"],
                        device.get("CgroupPermissions") or "rwm",
                    )
                ),
            )
        )
    for container_port, bindings in sorted((host.get("PortBindings") or {}).items()):
        for binding in bindings or []:
            host_ip = binding.get("HostIp") or ""
            host_port = binding.get("HostPort") or ""
            published = ":".join(value for value in (host_ip, host_port) if value)
            args.extend(("--publish", f"{published}:{container_port}"))

    if host.get("Privileged"):
        args.append("--privileged")
    if host.get("ReadonlyRootfs"):
        args.append("--read-only")
    if host.get("Init"):
        args.append("--init")
    for capability in host.get("CapAdd") or []:
        args.extend(("--cap-add", capability))
    for capability in host.get("CapDrop") or []:
        args.extend(("--cap-drop", capability))
    for option in host.get("SecurityOpt") or []:
        args.extend(("--security-opt", option))
    for group in host.get("GroupAdd") or []:
        args.extend(("--group-add", str(group)))
    for extra_host in host.get("ExtraHosts") or []:
        args.extend(("--add-host", extra_host))
    for dns in host.get("Dns") or []:
        args.extend(("--dns", dns))
    for search in host.get("DnsSearch") or []:
        args.extend(("--dns-search", search))
    for option in host.get("DnsOptions") or []:
        args.extend(("--dns-option", option))
    for key, value in sorted((host.get("Sysctls") or {}).items()):
        args.extend(("--sysctl", f"{key}={value}"))
    for limit in host.get("Ulimits") or []:
        args.extend(
            (
                "--ulimit",
                f"{limit['Name']}={limit['Soft']}:{limit['Hard']}",
            )
        )

    device_requests = host.get("DeviceRequests") or []
    if device_requests:
        gpu_all = len(device_requests) == 1 and device_requests[0].get("Count") == -1
        if not gpu_all:
            raise SwapError("only the standard all-GPU device request is supported")
        args.extend(("--gpus", "all"))
    if host.get("ShmSize"):
        args.extend(("--shm-size", str(host["ShmSize"])))
    if host.get("Memory"):
        args.extend(("--memory", str(host["Memory"])))
    if host.get("MemorySwap"):
        args.extend(("--memory-swap", str(host["MemorySwap"])))
    if host.get("NanoCpus"):
        args.extend(("--cpus", str(host["NanoCpus"] / 1_000_000_000)))
    if host.get("CpuShares"):
        args.extend(("--cpu-shares", str(host["CpuShares"])))
    _append_pair(args, "--cpuset-cpus", host.get("CpusetCpus"))
    _append_pair(args, "--ipc", host.get("IpcMode"))
    _append_pair(args, "--pid", host.get("PidMode"))
    _append_pair(args, "--uts", host.get("UTSMode"))

    log_config = host.get("LogConfig") or {}
    _append_pair(args, "--log-driver", log_config.get("Type"))
    for key, value in sorted((log_config.get("Config") or {}).items()):
        args.extend(("--log-opt", f"{key}={value}"))

    entrypoint = config.get("Entrypoint") or []
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    command = list(entrypoint[1:]) + list(config.get("Cmd") or [])
    if entrypoint:
        args.extend(("--entrypoint", entrypoint[0]))
    args.append(image)
    args.extend(str(item) for item in command)
    return args


def _camera_requirement_error(
    payload: dict[str, Any],
    required_fresh_cameras: tuple[str, ...],
    required_camera_backends: dict[str, str],
) -> str | None:
    cameras = {
        str(camera.get("id")): camera
        for camera in payload.get("cameras") or []
        if isinstance(camera, dict) and camera.get("id") is not None
    }
    for camera_id in required_fresh_cameras:
        camera = cameras.get(camera_id)
        if camera is None:
            return f"required camera {camera_id!r} is missing"
        if not camera.get("frameFresh"):
            return f"required camera {camera_id!r} is not fresh"
    for camera_id, expected_backend in required_camera_backends.items():
        camera = cameras.get(camera_id)
        if camera is None:
            return f"required camera {camera_id!r} is missing"
        connection = camera.get("connection") or {}
        if connection.get("captureBackend") != expected_backend:
            return f"required camera {camera_id!r} has the wrong capture backend"
    return None


def _wait_for_health(
    url: str,
    timeout_seconds: float,
    *,
    required_fresh_cameras: tuple[str, ...] = (),
    required_camera_backends: dict[str, str] | None = None,
) -> None:
    required_camera_backends = required_camera_backends or {}
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    if not required_fresh_cameras and not required_camera_backends:
                        return
                    try:
                        payload = json.load(response)
                    except (TypeError, ValueError):
                        last_error = "health response is not JSON"
                    else:
                        requirement_error = _camera_requirement_error(
                            payload,
                            required_fresh_cameras,
                            required_camera_backends,
                        )
                        if requirement_error is None:
                            return
                        last_error = requirement_error
                else:
                    last_error = f"HTTP {response.status}"
        except Exception as exc:  # Health failures are reported without secrets.
            last_error = type(exc).__name__
        time.sleep(1.0)
    raise SwapError(f"health check failed: {last_error}")


def _environment_matches(info: dict[str, Any], overrides: dict[str, str]) -> bool:
    current = dict(
        assignment.split("=", 1)
        for assignment in info["Config"].get("Env") or []
        if "=" in assignment
    )
    return all(current.get(key) == value for key, value in overrides.items())


def _runtime_matches(info: dict[str, Any], runtime_override: str | None) -> bool:
    return (
        runtime_override is None
        or info["HostConfig"].get("Runtime") == runtime_override
    )


def promote(
    *,
    active: str,
    image: str,
    rollback: str,
    health_url: str,
    health_timeout: float,
    env_overrides: dict[str, str],
    runtime_override: str | None,
    required_fresh_cameras: tuple[str, ...],
    required_camera_backends: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    info = _inspect_container(active)
    desired_image_id = _image_id(image)
    if (
        info.get("Image") == desired_image_id
        and _environment_matches(info, env_overrides)
        and _runtime_matches(info, runtime_override)
    ):
        if not dry_run:
            _wait_for_health(
                health_url,
                health_timeout,
                required_fresh_cameras=required_fresh_cameras,
                required_camera_backends=required_camera_backends,
            )
        return {"status": "already_promoted", "active": active, "image": image}
    if _container_exists(rollback):
        raise SwapError(f"rollback container {rollback!r} already exists")

    staging = f"{active}-swap-preflight-{os.getpid()}"
    create_args = build_create_args(
        info,
        name=staging,
        image=image,
        env_overrides=env_overrides,
        runtime_override=runtime_override,
    )
    if dry_run:
        return {
            "status": "dry_run",
            "active": active,
            "image": image,
            "rollback": rollback,
            "environment_overrides": sorted(env_overrides),
            "mounts": len(_volume_arguments(info)),
            "devices": len(info["HostConfig"].get("Devices") or []),
            "runtime": runtime_override or info["HostConfig"].get("Runtime"),
            "required_fresh_cameras": list(required_fresh_cameras),
            "required_camera_backends": dict(required_camera_backends),
        }

    candidate_created = False
    old_renamed = False
    try:
        _docker(*create_args)
        _docker("rm", staging)
        _docker("stop", "--time", "20", active)
        _docker("rename", active, rollback)
        old_renamed = True
        active_args = build_create_args(
            info,
            name=active,
            image=image,
            env_overrides=env_overrides,
            runtime_override=runtime_override,
        )
        _docker(*active_args)
        candidate_created = True
        _docker("start", active)
        _wait_for_health(
            health_url,
            health_timeout,
            required_fresh_cameras=required_fresh_cameras,
            required_camera_backends=required_camera_backends,
        )
    except BaseException:
        _docker("rm", "-f", staging, check=False)
        if candidate_created:
            _docker("rm", "-f", active, check=False)
        if old_renamed:
            _docker("rename", rollback, active, check=False)
            _docker("start", active, check=False)
            try:
                _wait_for_health(
                    health_url,
                    health_timeout,
                    required_fresh_cameras=required_fresh_cameras,
                    required_camera_backends=required_camera_backends,
                )
            except SwapError:
                pass
        raise
    return {
        "status": "promoted",
        "active": active,
        "image": image,
        "rollback": rollback,
        "health_url": health_url,
    }


def restore(
    *,
    active: str,
    rollback: str,
    displaced: str,
    health_url: str,
    health_timeout: float,
    required_fresh_cameras: tuple[str, ...],
    required_camera_backends: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    """Restore a preserved container and retain the displaced candidate."""
    _inspect_container(active)
    _inspect_container(rollback)
    if _container_exists(displaced):
        raise SwapError(f"displaced container {displaced!r} already exists")
    if dry_run:
        return {
            "status": "restore_dry_run",
            "active": active,
            "rollback": rollback,
            "displaced": displaced,
        }

    active_renamed = False
    rollback_renamed = False
    try:
        _docker("stop", "--time", "20", active)
        _docker("rename", active, displaced)
        active_renamed = True
        _docker("rename", rollback, active)
        rollback_renamed = True
        _docker("start", active)
        _wait_for_health(
            health_url,
            health_timeout,
            required_fresh_cameras=required_fresh_cameras,
            required_camera_backends=required_camera_backends,
        )
    except BaseException:
        if rollback_renamed:
            _docker("stop", "--time", "20", active, check=False)
            _docker("rename", active, rollback, check=False)
        if active_renamed:
            _docker("rename", displaced, active, check=False)
            _docker("start", active, check=False)
            try:
                _wait_for_health(
                    health_url,
                    health_timeout,
                    required_fresh_cameras=required_fresh_cameras,
                    required_camera_backends=required_camera_backends,
                )
            except SwapError:
                pass
        raise
    return {
        "status": "restored",
        "active": active,
        "rollback": rollback,
        "displaced": displaced,
        "health_url": health_url,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", required=True)
    parser.add_argument("--image")
    parser.add_argument("--rollback", required=True)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore --rollback and preserve the current active container.",
    )
    parser.add_argument(
        "--displaced",
        help="Name retained for the current active container during --restore.",
    )
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    parser.add_argument("--set-env", action="append", default=[])
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--runtime",
        help="Override the cloned OCI runtime, for example nvidia on Jetson.",
    )
    parser.add_argument(
        "--require-camera-fresh",
        action="append",
        default=[],
        metavar="CAMERA_ID",
        help="Wait for the named SafetyLens camera to report a fresh frame.",
    )
    parser.add_argument(
        "--require-camera-backend",
        action="append",
        default=[],
        metavar="CAMERA_ID=BACKEND",
        help="Wait for the named camera to report the expected capture backend.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.health_timeout <= 0:
        parser.error("health-timeout must be positive")
    if args.restore and not args.displaced:
        parser.error("--restore requires --displaced")
    if not args.restore and not args.image:
        parser.error("promotion requires --image")
    if args.restore and (args.image or args.set_env or args.env_file or args.runtime):
        parser.error("--restore cannot be combined with image or environment changes")
    try:
        required_fresh_cameras = tuple(dict.fromkeys(args.require_camera_fresh))
        required_camera_backends = _parse_camera_backends(args.require_camera_backend)
        if args.restore:
            result = restore(
                active=args.active,
                rollback=args.rollback,
                displaced=args.displaced,
                health_url=args.health_url,
                health_timeout=args.health_timeout,
                required_fresh_cameras=required_fresh_cameras,
                required_camera_backends=required_camera_backends,
                dry_run=args.dry_run,
            )
        else:
            overrides = _parse_env(args.set_env, args.env_file)
            result = promote(
                active=args.active,
                image=args.image,
                rollback=args.rollback,
                health_url=args.health_url,
                health_timeout=args.health_timeout,
                env_overrides=overrides,
                runtime_override=args.runtime,
                required_fresh_cameras=required_fresh_cameras,
                required_camera_backends=required_camera_backends,
                dry_run=args.dry_run,
            )
    except (OSError, SwapError) as exc:
        print(f"swap failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
