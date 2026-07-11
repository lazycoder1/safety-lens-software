#!/usr/bin/env python3
"""SSH-facing Rakshak Lens site configuration tool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import site_config


def _print_result(result: site_config.SiteConfigResult, *, json_output: bool = False) -> int:
    if json_output:
        print(site_config.result_to_json(result))
    else:
        status = "OK" if result.ok else "FAILED"
        print(f"[{status}] {result.source_path or site_config.default_site_config_path()}")
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        for warning in result.warnings:
            print(f"WARN: {warning}")
    return 0 if result.ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    result = site_config.load_site_config(args.config, strict_env=not args.allow_missing_env)
    return _print_result(result, json_output=args.json)


def cmd_plan(args: argparse.Namespace) -> int:
    result = site_config.build_plan(args.config)
    if args.json:
        print(site_config.result_to_json(result))
    elif result.ok and result.config:
        for line in result.config["summary"]:
            print(line)
    else:
        _print_result(result)
    return 0 if result.ok else 1


def cmd_apply(args: argparse.Namespace) -> int:
    plan = site_config.build_plan(args.config)
    if not plan.ok:
        return _print_result(plan, json_output=args.json)
    if not args.yes:
        if plan.config:
            for line in plan.config["summary"]:
                print(line)
        print("")
        print("Use --yes to apply this site config.")
        return 2

    if args.restart:
        _stop_service(args.service_name)
    result = site_config.apply_site_config(args.config)
    if args.restart:
        _start_service(args.service_name)
    if not result.ok:
        return _print_result(result, json_output=args.json)
    return _print_result(result, json_output=args.json)


def cmd_export(args: argparse.Namespace) -> int:
    target = site_config.export_site_config(args.output or args.config, redacted=not args.include_secrets)
    print(f"Exported site config to {target}")
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    from camera_discovery import discovery_to_site_document, discover_cameras

    cidrs = args.cidrs or None
    capability_model_overrides = _parse_capability_model_overrides(args.capability_model_override)
    if capability_model_overrides is None:
        return 2
    discovery = discover_cameras(cidrs, timeout_seconds=args.timeout_seconds)
    if args.json and (args.yaml or args.output_site_yaml):
        print("ERROR: --json cannot be combined with --yaml or --output-site-yaml", file=sys.stderr)
        return 2
    if args.yaml or args.output_site_yaml:
        doc = discovery_to_site_document(
            discovery,
            site_name=args.site_name,
            timezone=args.timezone,
            merge_existing=not args.replace_existing,
            zone=args.zone,
            profile=args.profile,
            capabilities=args.capabilities,
            capability_model_overrides=capability_model_overrides,
            safety_rule_ids=args.safety_rule_ids,
            username_env=args.username_env,
            password_env=args.password_env,
            default_stream_path=args.default_stream_path,
            fps=args.fps,
            enabled=args.enable_cameras,
            include_active_windows=not args.no_active_windows,
            include_event_policy=not args.no_event_policy,
            event_output_ids=args.event_output_ids,
            event_severity=args.event_severity,
            event_priority=args.event_priority,
            event_cooldown_seconds=args.event_cooldown_seconds,
            event_min_confidence=args.event_min_confidence,
            event_message_template=args.event_message_template,
        )
        rendered = yaml.safe_dump(doc, sort_keys=False, allow_unicode=False)
        if args.output_site_yaml:
            target = Path(args.output_site_yaml)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
            if not args.yaml:
                print(f"Wrote discovered camera site YAML to {target}")
                print(f"Discovered {len(doc.get('cameras', {}))} camera candidate(s)")
                if not args.enable_cameras:
                    print("Cameras are disabled in the generated YAML; pass --enable-cameras after stream paths and credentials are verified.")
        if args.yaml:
            print(rendered, end="")
        return 0
    if args.json:
        print(json.dumps(discovery, indent=2))
    else:
        print(f"CIDRs: {', '.join(discovery.get('cidrs') or cidrs or ['auto'])}")
        for warning in discovery.get("warnings", []):
            print(f"WARN: {warning}")
        devices = discovery.get("devices", [])
        print(f"Found {len(devices)} camera candidate(s)")
        for device in devices:
            host = device.get("host") or device.get("ip") or "unknown-host"
            name = device.get("name") or device.get("vendor") or "Camera"
            path = device.get("stream_path") or ""
            print(f"- {host} {name} {path}".rstrip())
    return 0


def _parse_capability_model_overrides(values: list[str] | None) -> dict[str, str] | None:
    overrides: dict[str, str] = {}
    for raw_value in values or []:
        if "=" not in raw_value:
            print(
                "ERROR: --capability-model-override must use capability=model_key",
                file=sys.stderr,
            )
            return None
        capability, model_key = raw_value.split("=", 1)
        capability = capability.strip()
        model_key = model_key.strip()
        if not capability or not model_key:
            print(
                "ERROR: --capability-model-override must use non-empty capability=model_key",
                file=sys.stderr,
            )
            return None
        overrides[capability] = model_key
    return overrides


def cmd_doctor(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    if args.backend_url:
        env["SAFETYLENS_BACKEND_URL"] = args.backend_url
    if args.frontend_url:
        env["SAFETYLENS_FRONTEND_URL"] = args.frontend_url
    if args.model_server_url:
        env["SAFETYLENS_MODEL_SERVER_URL"] = args.model_server_url
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "runtime_doctor.py")], env=env)


def _stop_service(service_name: str) -> None:
    if not service_name:
        return
    if not _has_command("systemctl"):
        print("WARN: systemctl not found; applying config without service stop")
        return
    _run_service_action("stop", service_name)


def _start_service(service_name: str) -> None:
    if not service_name:
        return
    if not _has_command("systemctl"):
        print("WARN: systemctl not found; config applied but service was not restarted")
        return
    _run_service_action("start", service_name)


def _run_service_action(action: str, service_name: str) -> None:
    cmd = ["systemctl", action, service_name]
    if os.geteuid() != 0 and _has_command("sudo"):
        cmd.insert(0, "sudo")
    verb = {"stop": "Stopping", "start": "Starting"}.get(action, action.capitalize())
    print(f"{verb} {service_name}...")
    subprocess.run(cmd, check=True)


def _has_command(name: str) -> bool:
    return subprocess.call(
        ["sh", "-c", f"command -v {name} >/dev/null 2>&1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, plan, apply, export, discover, and doctor Rakshak Lens site configuration."
    )
    parser.add_argument(
        "--config",
        default=str(site_config.default_site_config_path()),
        help="Path to site YAML. Default: /etc/safetylens/site.yaml or SAFETYLENS_SITE_CONFIG.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate site YAML without changing runtime config.")
    validate.add_argument("--allow-missing-env", action="store_true", help="Warn instead of failing for missing ${ENV} refs.")
    validate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    validate.set_defaults(func=cmd_validate)

    plan = sub.add_parser("plan", help="Show changes between active runtime config and site YAML.")
    plan.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    plan.set_defaults(func=cmd_plan)

    apply = sub.add_parser("apply", help="Apply site YAML into the active config store.")
    apply.add_argument("--yes", action="store_true", help="Apply changes after showing the plan.")
    apply.add_argument("--restart", action="store_true", help="Restart backend systemd service after apply.")
    apply.add_argument("--service-name", default="rakshak-lens-backend", help="Systemd service to restart.")
    apply.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    apply.set_defaults(func=cmd_apply)

    export = sub.add_parser("export", help="Export active runtime config back to site YAML.")
    export.add_argument("--output", help="Output YAML path. Defaults to --config.")
    export.add_argument("--include-secrets", action="store_true", help="Export stored secrets instead of redacted values.")
    export.set_defaults(func=cmd_export)

    discover = sub.add_parser("discover", help="Discover ONVIF/RTSP cameras on the local network.")
    discover.add_argument("cidrs", nargs="*", help="Optional CIDR ranges, e.g. 192.168.1.0/24.")
    discover.add_argument("--timeout-seconds", type=float, default=4.0)
    discover.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    discover.add_argument("--yaml", action="store_true", help="Print discovered cameras as site YAML.")
    discover.add_argument("--output-site-yaml", help="Write discovered cameras as site YAML to this path.")
    discover.add_argument("--site-name", default="Rakshak Lens Discovered Cameras")
    discover.add_argument("--timezone", default="Asia/Kolkata")
    discover.add_argument("--replace-existing", action="store_true", help="Set site.merge_existing=false in generated YAML.")
    discover.add_argument("--zone", default="Discovered Cameras", help="Default zone assigned to generated cameras.")
    discover.add_argument("--profile", default="general_safety", help="Default camera profile.")
    discover.add_argument(
        "--capabilities",
        nargs="+",
        help="Capabilities assigned to generated cameras, e.g. person_presence vehicle_presence.",
    )
    discover.add_argument(
        "--capability-model-override",
        action="append",
        default=[],
        metavar="CAPABILITY=MODEL_KEY",
        help=(
            "Optional per-capability model override for generated cameras, "
            "e.g. apron_required=ppe_closed_set_candidate. Repeat for multiple capabilities."
        ),
    )
    discover.add_argument(
        "--safety-rule-ids",
        nargs="*",
        help="Optional safety rule IDs assigned to generated cameras. Defaults are inferred from capabilities.",
    )
    discover.add_argument("--username-env", default="", help="Environment variable name to reference for RTSP usernames.")
    discover.add_argument("--password-env", default="", help="Environment variable name to reference for RTSP passwords.")
    discover.add_argument("--default-stream-path", default="/Streaming/Channels/101")
    discover.add_argument("--fps", type=int, default=6)
    discover.add_argument("--enable-cameras", action="store_true", help="Enable generated cameras immediately.")
    discover.add_argument("--no-active-windows", action="store_true", help="Do not add all-week capability active windows.")
    discover.add_argument("--no-event-policy", action="store_true", help="Do not add per-camera event_policy blocks.")
    discover.add_argument(
        "--event-output-ids",
        nargs="*",
        default=["in_app"],
        help="Alert output IDs for generated per-camera event policies. Default: in_app.",
    )
    discover.add_argument("--event-severity", default="inherit", help="Severity for generated event policies: inherit, P1, P2, P3, or P4.")
    discover.add_argument("--event-priority", type=int, default=5, help="Priority for generated event policies.")
    discover.add_argument("--event-cooldown-seconds", type=int, default=60, help="Cooldown for generated event policies.")
    discover.add_argument("--event-min-confidence", type=float, default=0.0, help="Minimum confidence condition for generated event policies.")
    discover.add_argument(
        "--event-message-template",
        default="{severity} {violation_type} on {camera} in {zone}",
        help="Message template for generated event policies.",
    )
    discover.set_defaults(func=cmd_discover)

    doctor = sub.add_parser("doctor", help="Run runtime health checks after applying config.")
    doctor.add_argument("--backend-url")
    doctor.add_argument("--frontend-url")
    doctor.add_argument("--model-server-url")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
