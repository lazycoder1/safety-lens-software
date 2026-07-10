#!/usr/bin/env python3
"""Read-only validation of Jetson nvpmodel profile and runtime state."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Sequence

CPU_REFERENCE_RE = re.compile(r"^CORE_(\d+)\s+(\S+)", re.MULTILINE)
MODE_RE = re.compile(r"Current mode:\s*NV Power Mode:\s*(.+)")


def parse_cpu_list(value: str) -> list[int]:
    """Parse Linux CPU-list syntax such as ``0-3,5``."""
    cpus: set[int] = set()
    for item in value.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            cpus.add(int(item))
            continue
        start_text, end_text = item.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid CPU range: {item}")
        cpus.update(range(start, end + 1))
    return sorted(cpus)


def profile_cpu_references(path: Path) -> dict[int, str]:
    """Return CPU IDs and sysfs paths referenced by an nvpmodel profile."""
    content = path.read_text(encoding="utf-8")
    return {int(cpu_id): sysfs_path for cpu_id, sysfs_path in CPU_REFERENCE_RE.findall(content)}


def compatible_profiles(profile_dir: Path, present_cpus: Sequence[int]) -> list[str]:
    """List installed profiles that do not reference nonexistent CPUs."""
    present = set(present_cpus)
    candidates = []
    for path in sorted(profile_dir.glob("nvpmodel_*.conf")):
        try:
            references = profile_cpu_references(path)
        except (OSError, UnicodeError):
            continue
        if references and set(references) == present:
            candidates.append(str(path))
    return candidates


def _read_cpu_frequency(sys_root: Path, cpu_id: int) -> int | None:
    path = sys_root / f"devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_max_freq"
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_gpu_frequency(sys_root: Path) -> int | None:
    candidates = (
        sys_root / "devices/17000000.ga10b/devfreq_dev/max_freq",
        sys_root / "devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq",
    )
    for path in candidates:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return None


def _run_nvpmodel_query(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)


def build_report(
    *,
    nvpmodel_conf: Path = Path("/etc/nvpmodel.conf"),
    sys_root: Path = Path("/sys"),
    nvpmodel_bin: Path = Path("/usr/sbin/nvpmodel"),
    require_all_cpus: bool = False,
    require_mode: str | None = None,
    query_runner: Callable[[Sequence[str]], subprocess.CompletedProcess] = _run_nvpmodel_query,
) -> dict:
    """Inspect an nvpmodel profile and runtime without changing device state."""
    issues: list[dict] = []
    present_path = sys_root / "devices/system/cpu/present"
    online_path = sys_root / "devices/system/cpu/online"

    try:
        present_cpus = parse_cpu_list(present_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "supported": False,
            "issues": [{"code": "cpu_topology_unavailable", "detail": str(exc)}],
        }

    try:
        online_cpus = parse_cpu_list(online_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        online_cpus = []
        issues.append({"code": "online_cpu_state_unavailable", "detail": str(exc)})

    try:
        resolved_profile = nvpmodel_conf.resolve(strict=True)
        references = profile_cpu_references(resolved_profile)
    except (OSError, UnicodeError) as exc:
        return {
            "ok": False,
            "supported": True,
            "presentCpuIds": present_cpus,
            "onlineCpuIds": online_cpus,
            "profile": str(nvpmodel_conf),
            "issues": [{"code": "nvpmodel_profile_unavailable", "detail": str(exc)}],
        }

    invalid_references = [cpu_id for cpu_id in sorted(references) if cpu_id not in present_cpus]
    if invalid_references:
        issues.append({
            "code": "profile_references_missing_cpus",
            "detail": f"Profile references nonexistent CPU IDs: {invalid_references}",
            "cpuIds": invalid_references,
        })

    query_output = ""
    query_return_code = None
    active_mode = None
    try:
        query = query_runner([str(nvpmodel_bin), "-q", "--verbose", "-f", str(resolved_profile)])
        query_return_code = query.returncode
        query_output = "\n".join(part for part in (query.stdout, query.stderr) if part).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        issues.append({"code": "nvpmodel_query_failed", "detail": str(exc)})
    else:
        match = MODE_RE.search(query_output)
        if match:
            active_mode = match.group(1).strip()
        if query_return_code != 0:
            issues.append({
                "code": "nvpmodel_query_failed",
                "detail": f"nvpmodel exited with status {query_return_code}",
            })
        elif "power mode is not set" in query_output.lower() or active_mode is None:
            issues.append({"code": "power_mode_unset", "detail": "nvpmodel has no active power mode"})

    offline_cpus = sorted(set(present_cpus) - set(online_cpus))
    if require_all_cpus and offline_cpus:
        issues.append({
            "code": "required_cpus_offline",
            "detail": f"Required CPU IDs are offline: {offline_cpus}",
            "cpuIds": offline_cpus,
        })
    if require_mode and (active_mode or "").lower() != require_mode.lower():
        issues.append({
            "code": "required_power_mode_inactive",
            "detail": f"Expected {require_mode}, found {active_mode or 'unset'}",
            "expected": require_mode,
            "actual": active_mode,
        })

    candidates = compatible_profiles(resolved_profile.parent, present_cpus) if invalid_references else []
    recommendations = []
    if invalid_references:
        recommendations.append(
            "Select an installed nvpmodel profile matching the physical module SKU; do not choose from CPU count alone."
        )
    if any(issue["code"] == "power_mode_unset" for issue in issues):
        recommendations.append("Set an explicit, thermally validated nvpmodel mode before performance testing.")

    return {
        "ok": not issues,
        "supported": True,
        "profile": str(nvpmodel_conf),
        "resolvedProfile": str(resolved_profile),
        "presentCpuIds": present_cpus,
        "onlineCpuIds": online_cpus,
        "offlineCpuIds": offline_cpus,
        "profileCpuIds": sorted(references),
        "invalidProfileCpuIds": invalid_references,
        "compatibleProfileCandidates": candidates,
        "activeMode": active_mode,
        "queryReturnCode": query_return_code,
        "cpuMaxFrequencyKHz": {
            str(cpu_id): frequency
            for cpu_id in present_cpus
            if (frequency := _read_cpu_frequency(sys_root, cpu_id)) is not None
        },
        "gpuMaxFrequencyHz": _read_gpu_frequency(sys_root),
        "requirements": {
            "allCpusOnline": require_all_cpus,
            "mode": require_mode,
        },
        "issues": issues,
        "recommendations": recommendations,
    }


def _human_summary(report: dict) -> str:
    lines = [f"Jetson power profile: {'PASS' if report.get('ok') else 'FAIL'}"]
    if not report.get("supported"):
        lines.append("CPU topology is unavailable; this does not look like a supported Linux target.")
    else:
        lines.extend([
            f"Profile: {report.get('resolvedProfile', report.get('profile', 'unknown'))}",
            f"Mode: {report.get('activeMode') or 'unset'}",
            f"CPUs: online {report.get('onlineCpuIds', [])} / present {report.get('presentCpuIds', [])}",
            f"GPU max: {report.get('gpuMaxFrequencyHz') or 'unknown'} Hz",
        ])
    for issue in report.get("issues", []):
        lines.append(f"- {issue['code']}: {issue['detail']}")
    for recommendation in report.get("recommendations", []):
        lines.append(f"Next: {recommendation}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvpmodel-conf", type=Path, default=Path("/etc/nvpmodel.conf"))
    parser.add_argument("--sys-root", type=Path, default=Path("/sys"))
    parser.add_argument("--nvpmodel-bin", type=Path, default=Path("/usr/sbin/nvpmodel"))
    parser.add_argument("--require-all-cpus", action="store_true")
    parser.add_argument("--require-mode")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    report = build_report(
        nvpmodel_conf=args.nvpmodel_conf,
        sys_root=args.sys_root,
        nvpmodel_bin=args.nvpmodel_bin,
        require_all_cpus=args.require_all_cpus,
        require_mode=args.require_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True) if args.as_json else _human_summary(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
