#!/usr/bin/env python3
"""Generate a signed SafetyLens license for local development and testing.

Until License Hub is built, this script is the only way to produce a valid
.lic file. Reads the private key from ``~/.safetylens/license_private.pem``
(must be the matching pair to ``backend/keys/license_pub.pem``).

Examples
--------

Issue a default 1-year license for 10 cameras::

    python scripts/generate_test_license.py \\
        --customer "TMEIC Jamshedpur" \\
        --cameras 10 \\
        --out tmeic.lic

Issue a license that expires in 60 seconds for testing the suspension flow::

    python scripts/generate_test_license.py \\
        --customer "Expiry Test" \\
        --cameras 2 \\
        --expires-in-seconds 60 \\
        --out expiring.lic
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

DEFAULT_PRIVATE_KEY = Path.home() / ".safetylens" / "license_private.pem"
SCHEMA_VERSION = 1


def canonicalize(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--customer", required=True, help="Customer name")
    parser.add_argument("--cameras", type=int, default=10, help="Max cameras (default: 10)")
    parser.add_argument(
        "--features",
        default="base",
        help="Comma-separated feature list (default: base)",
    )
    parser.add_argument("--license-id", help="License ID (default: auto-generated SL-YYYY-XXXX)")
    parser.add_argument("--issued-by", default="techser", help="Partner slug (default: techser)")
    parser.add_argument(
        "--heartbeat-url",
        default="http://localhost:3000/api/heartbeat",
        help="License Hub heartbeat endpoint (default: http://localhost:3000/api/heartbeat)",
    )
    parser.add_argument(
        "--expires-in-days", type=int, default=365, help="Days until expiry (default: 365)"
    )
    parser.add_argument(
        "--expires-in-seconds",
        type=int,
        help="Override: seconds until expiry. Useful for testing suspension flow.",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=DEFAULT_PRIVATE_KEY,
        help=f"Ed25519 private key PEM (default: {DEFAULT_PRIVATE_KEY})",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .lic file path")
    args = parser.parse_args()

    if not args.private_key.is_file():
        print(f"ERROR: Private key not found at {args.private_key}", file=sys.stderr)
        print(
            "Generate one with: openssl genpkey -algorithm Ed25519 -out ~/.safetylens/license_private.pem",
            file=sys.stderr,
        )
        return 1

    private_key = serialization.load_pem_private_key(args.private_key.read_bytes(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        print(f"ERROR: {args.private_key} is not an Ed25519 private key", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc).replace(microsecond=0)
    if args.expires_in_seconds is not None:
        expires = now + timedelta(seconds=args.expires_in_seconds)
    else:
        expires = now + timedelta(days=args.expires_in_days)

    license_id = args.license_id or f"SL-{now.year}-{int(now.timestamp()) % 10000:04d}"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "license_id": license_id,
        "customer_name": args.customer,
        "issued_by_partner": args.issued_by,
        "max_cameras": args.cameras,
        "features": [f.strip() for f in args.features.split(",") if f.strip()],
        "issued_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "heartbeat_url": args.heartbeat_url,
    }
    canonical = canonicalize(payload)
    signature = private_key.sign(canonical)
    signed = {**payload, "signature": base64.b64encode(signature).decode("ascii")}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(signed, indent=2))
    print(f"License written to {args.out}")
    print(f"  ID: {license_id}")
    print(f"  Customer: {args.customer}")
    print(f"  Cameras: {args.cameras}")
    print(f"  Features: {payload['features']}")
    print(f"  Expires: {expires.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
