# SafetyLens Licensing Spec

This is the **single source of truth** for the license file format, the heartbeat token format, the signing/verification algorithm, and the enforcement state machine. Both SafetyLens edge (`video-analytics`) and License Hub (separate repo) implement against this spec.

## Goals

1. Edge devices verify licenses **offline** — signature + expiry check, no network call required for the primary `.lic` file.
2. Techser can **revoke** a license remotely by stopping heartbeat issuance — within ~6 weeks of the revoke decision the edge device suspends inference.
3. Implementation partners can generate licenses self-service via a web UI without ever touching the private key.
4. Tampering is impossible — flip one byte in a license file and verification fails.
5. The edge admin UI **never** locks out the customer — they can always upload a fresh license, even when inference is suspended.

## Cryptography

- Algorithm: **Ed25519** (RFC 8032)
- Library on edge: Python `cryptography` package
- Library on License Hub: Node `@noble/ed25519` or `node:crypto`
- Key encoding: PEM (PKCS#8 for private, SubjectPublicKeyInfo for public)
- Signature encoding: base64-encoded raw 64-byte signature
- Canonicalization: payload is JSON-serialized with **sorted keys**, **no whitespace**, **UTF-8** before signing — both sides must use the exact same canonicalization or signatures won't match

## License File (`.lic`)

The long-lived license. Issued once at deployment, valid for the contract duration (typically 1 year). Does not change unless the customer renews or upgrades.

### Schema

```json
{
  "schema_version": 1,
  "license_id": "SL-2026-0001",
  "customer_name": "TMEIC Jamshedpur",
  "issued_by_partner": "techser",
  "max_cameras": 10,
  "features": ["base"],
  "issued_at": "2026-04-12T00:00:00Z",
  "expires_at": "2027-04-12T00:00:00Z",
  "heartbeat_url": "https://licenses.techser.com/api/heartbeat",
  "signature": "base64(ed25519_sign(canonical_payload_without_signature_field))"
}
```

### Field reference

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Bump if we ever change the format. Edge refuses unknown versions. |
| `license_id` | string | Format: `SL-YYYY-NNNN`. Globally unique. Used as the join key in heartbeat tokens. |
| `customer_name` | string | Free-form. Shown on the License page in the edge UI. |
| `issued_by_partner` | string | Partner slug from License Hub. Audit trail. |
| `max_cameras` | int | Hard cap enforced when adding cameras. |
| `features` | string[] | V1 only supports `["base"]`. Future: `anpr`, `face`, `ai_search`. Edge ignores unknown values. |
| `issued_at` | ISO 8601 UTC | Signed start of the initial heartbeat window. Until the first heartbeat is stored, enforcement uses `issued_at + 35 days`. |
| `expires_at` | ISO 8601 UTC | Hard expiry. After this + grace period, inference suspends. |
| `heartbeat_url` | string | Where the edge fetches refresh tokens. Lets us migrate Hub URLs without re-issuing licenses. |
| `signature` | string (base64) | Ed25519 signature over the canonical payload with `signature` field removed. |

### Signing algorithm (License Hub side)

```python
import json, base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

def sign_license(payload: dict, private_key: Ed25519PrivateKey) -> dict:
    payload_to_sign = {k: v for k, v in payload.items() if k != "signature"}
    canonical = json.dumps(
        payload_to_sign,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    ).encode("utf-8")
    sig = private_key.sign(canonical)
    return {**payload_to_sign, "signature": base64.b64encode(sig).decode("ascii")}
```

### Verification algorithm (edge side)

```python
def verify_license(blob: bytes, public_key: Ed25519PublicKey) -> dict:
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("schema_version") != 1:
        raise InvalidLicense("unsupported schema version")
    sig_b64 = payload.pop("signature", None)
    if not sig_b64:
        raise InvalidLicense("missing signature")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    ).encode("utf-8")
    public_key.verify(base64.b64decode(sig_b64), canonical)  # raises InvalidSignature
    return payload
```

## Heartbeat Token

A short-lived token the edge fetches from License Hub once a month. This is the real revocation mechanism — stop issuing heartbeats and the edge stops working after the grace period.

### Schema

```json
{
  "schema_version": 1,
  "license_id": "SL-2026-0001",
  "issued_at": "2026-04-12T00:00:00Z",
  "valid_until": "2026-05-17T00:00:00Z",
  "signature": "base64(ed25519_sign(canonical_payload_without_signature_field))"
}
```

### Field reference

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Same as license. |
| `license_id` | string | Must match the installed license, otherwise rejected. |
| `issued_at` | ISO 8601 UTC | When License Hub signed this token. |
| `valid_until` | ISO 8601 UTC | 35 days after issue. Edge stops trusting after this + grace period. |
| `signature` | string (base64) | Ed25519 over canonical payload, same algorithm as license. |

35 days gives a 5-day buffer over the 30-day refresh interval. If a refresh fails, the edge has up to 5 days to retry before any warning kicks in.

## Enforcement State Machine

The edge computes a single `LicenseState` from the combination of (`.lic` expiry, heartbeat `valid_until`, current time, grace period).

### States

| State | Trigger | Inference | UI Banner | Admin UI |
|---|---|---|---|---|
| `valid` | Both license and heartbeat windows are valid and license expiry is more than 14 days away | runs | none | reachable |
| `warning` | License is within 14 days of expiry, while neither license nor heartbeat is overdue | runs | yellow: "License expires in N days" | reachable |
| `grace` | License is past `expires_at` but within its 14-day grace, OR heartbeat is past `valid_until` but within its 14-day grace | runs | red: "N days until suspension" | reachable |
| `suspended` | Past grace period on either license or heartbeat | **stopped** | red: "License suspended — please upload a new license" | **still reachable** |

### Transitions

- Re-evaluated at backend startup
- Re-evaluated on every license/heartbeat upload
- Re-evaluated by a periodic check every 1 hour
- When the state transitions to `suspended`, the inference workers are signalled to stop and any new camera frames are dropped
- When a fresh license or heartbeat takes the state back to `valid`, inference workers are restarted automatically

### Grace period

**14 days** for both:
- License `expires_at` past → 14 days of `grace` → then `suspended`
- Heartbeat `valid_until` past → 14 days of `grace` → then `suspended`

The `valid_until` is set 35 days after issuance, so a healthy edge with a working refresh job will never enter the warning state. The grace period only matters if the refresh job fails.

Before the first heartbeat exists, the edge derives `valid_until` from the
signed license as `license.issued_at + 35 days`. The following 14-day grace is
the final deadline. Removing a heartbeat file therefore cannot reset the clock.

## Heartbeat Refresh Loop (edge side)

- Background asyncio task started by FastAPI
- Runs once a day (24h interval, with jitter)
- Calls `POST {heartbeat_url}` from the license file with body `{"license_id": "...", "current_heartbeat_signature": "..."}`
- On success: writes the new token to `backend/heartbeat/current.json` and reloads
- On failure: logs the error, retries the next day. No exponential backoff — once a day is fine.
- Air-gapped customers: the same heartbeat token can be uploaded manually via the License page in the admin UI.

## File Layout on Edge

```
/app/backend/keys/
  license_pub.pem                    # Ed25519 public key, baked into image
/var/lib/safetylens/license/
  current.lic                        # Active license in persistent volume
/var/lib/safetylens/heartbeat/
  current.json                       # Active heartbeat in persistent volume
```

`/var/lib/safetylens` is a stable named Docker volume in production so license
and heartbeat state survive container recreation and image upgrades. See
[`runtime-state.md`](runtime-state.md) for the migration and verification gate.

## Versioning

Both schemas have a `schema_version: 1` field. If we ever need to evolve the format:

1. Edge accepts both `1` and `2`
2. Hub starts issuing `2`
3. Old `1` licenses keep working until they expire
4. After all `1` licenses have expired, edge drops support for `1`

This avoids ever needing to forcibly re-issue licenses in the field.

## Open Questions / Future Work

- **Multi-key trust store** for graceful key rotation — not in v1
- **Per-camera feature flags** (e.g., "ANPR enabled on cameras 1-3 only") — not in v1, current model is per-deployment
- **Hardware binding** — explicitly out of scope per product decision
- **Offline grace via signed offline tokens** — partner can pre-generate a 6-month "offline pack" of heartbeat tokens for truly air-gapped deployments. Maybe v2.
