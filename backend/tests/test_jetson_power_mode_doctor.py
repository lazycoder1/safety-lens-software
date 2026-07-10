import importlib.util
from pathlib import Path
from subprocess import CompletedProcess


ROOT = Path(__file__).resolve().parents[2]
DOCTOR_PATH = ROOT / "scripts" / "jetson_power_mode_doctor.py"
SPEC = importlib.util.spec_from_file_location("jetson_power_mode_doctor", DOCTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _profile(path: Path, cpu_ids: list[int]) -> None:
    lines = ["< PARAM TYPE=FILE NAME=CPU_ONLINE >"]
    lines.extend(f"CORE_{cpu_id} /sys/devices/system/cpu/cpu{cpu_id}/online" for cpu_id in cpu_ids)
    lines.extend(["", "< POWER_MODEL ID=0 NAME=MAXN >"])
    lines.extend(f"CPU_ONLINE CORE_{cpu_id} 1" for cpu_id in cpu_ids)
    path.write_text("\n".join(lines))


def _sysfs(tmp_path: Path, *, present: str, online: str) -> Path:
    sys_root = tmp_path / "sys"
    _write(sys_root / "devices/system/cpu/present", present)
    _write(sys_root / "devices/system/cpu/online", online)
    for cpu_id in doctor.parse_cpu_list(present):
        _write(
            sys_root / f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_max_freq",
            "1984000\n",
        )
    _write(sys_root / "devices/17000000.ga10b/devfreq_dev/max_freq", "765000000\n")
    return sys_root


def _query(output: str, returncode: int = 0):
    return lambda _command: CompletedProcess([], returncode, stdout=output, stderr="")


def test_parse_cpu_list_supports_ranges_and_individual_ids():
    assert doctor.parse_cpu_list("0-3,5,7-8\n") == [0, 1, 2, 3, 5, 7, 8]


def test_wrong_sku_profile_and_unset_mode_are_reported(tmp_path):
    sys_root = _sysfs(tmp_path, present="0-5\n", online="0-3\n")
    profile_dir = tmp_path / "etc/nvpmodel"
    profile_dir.mkdir(parents=True)
    wrong_profile = profile_dir / "nvpmodel_p3767_0000.conf"
    compatible_profile = profile_dir / "nvpmodel_p3767_0001.conf"
    _profile(wrong_profile, list(range(8)))
    _profile(compatible_profile, list(range(6)))
    active_link = tmp_path / "etc/nvpmodel.conf"
    active_link.symlink_to(wrong_profile)

    report = doctor.build_report(
        nvpmodel_conf=active_link,
        sys_root=sys_root,
        query_runner=_query("NVPM WARN: power mode is not set!"),
    )

    assert not report["ok"]
    assert report["invalidProfileCpuIds"] == [6, 7]
    assert report["offlineCpuIds"] == [4, 5]
    assert report["compatibleProfileCandidates"] == [str(compatible_profile)]
    assert {issue["code"] for issue in report["issues"]} == {
        "profile_references_missing_cpus",
        "power_mode_unset",
    }


def test_matching_maxn_profile_passes_required_capacity_gate(tmp_path):
    sys_root = _sysfs(tmp_path, present="0-5\n", online="0-5\n")
    profile = tmp_path / "etc/nvpmodel/nvpmodel_p3767_0001.conf"
    profile.parent.mkdir(parents=True)
    _profile(profile, list(range(6)))

    report = doctor.build_report(
        nvpmodel_conf=profile,
        sys_root=sys_root,
        require_all_cpus=True,
        require_mode="MAXN",
        query_runner=_query("NVPM VERB: Current mode: NV Power Mode: MAXN\n0\n"),
    )

    assert report["ok"]
    assert report["activeMode"] == "MAXN"
    assert report["onlineCpuIds"] == list(range(6))
    assert report["gpuMaxFrequencyHz"] == 765000000
    assert report["cpuMaxFrequencyKHz"] == {str(cpu_id): 1984000 for cpu_id in range(6)}


def test_capacity_requirements_fail_for_valid_lower_power_mode(tmp_path):
    sys_root = _sysfs(tmp_path, present="0-5\n", online="0-3\n")
    profile = tmp_path / "etc/nvpmodel/nvpmodel_p3767_0001.conf"
    profile.parent.mkdir(parents=True)
    _profile(profile, list(range(6)))

    report = doctor.build_report(
        nvpmodel_conf=profile,
        sys_root=sys_root,
        require_all_cpus=True,
        require_mode="MAXN",
        query_runner=_query("NVPM VERB: Current mode: NV Power Mode: 15W\n2\n"),
    )

    assert not report["ok"]
    assert {issue["code"] for issue in report["issues"]} == {
        "required_cpus_offline",
        "required_power_mode_inactive",
    }


def test_query_failure_is_visible(tmp_path):
    sys_root = _sysfs(tmp_path, present="0-1\n", online="0-1\n")
    profile = tmp_path / "etc/nvpmodel/nvpmodel_test.conf"
    profile.parent.mkdir(parents=True)
    _profile(profile, [0, 1])

    report = doctor.build_report(
        nvpmodel_conf=profile,
        sys_root=sys_root,
        query_runner=_query("permission denied", returncode=1),
    )

    assert not report["ok"]
    assert report["issues"] == [{
        "code": "nvpmodel_query_failed",
        "detail": "nvpmodel exited with status 1",
    }]
