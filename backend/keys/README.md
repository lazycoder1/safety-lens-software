# Licensing Keys

This folder holds the **public key** used by SafetyLens edge to verify signed license files and heartbeat tokens issued by License Hub.

## Files

| File | Status | Purpose |
|---|---|---|
| `license_pub.pem` | **committed** | Ed25519 public key. Loaded at backend startup to verify `.lic` files and heartbeat tokens. |
| `license_priv.pem` | **never committed** (gitignored) | Would only ever live here on a developer machine for local testing. Production private key lives in License Hub env vars. |

## Where the private key actually lives

- **Developer machine (Gautham):** `~/.safetylens/license_private.pem` (mode 600)
- **Production:** Vercel encrypted env var `LICENSE_SIGNING_KEY` on the License Hub deployment
- **Backup:** Sealed copy in 1Password vault `Techser → SafetyLens → License Signing Key`

## Rotating the key

Rotating the Ed25519 keypair is a breaking change — every existing `.lic` file in the field would stop verifying. Don't do it casually. The procedure is:

1. Generate new keypair
2. Ship new public key in next SafetyLens release
3. Re-issue every active license signed with the new private key
4. Distribute the new `.lic` files to all customers

For now we treat the keypair as permanent. If we ever need rotation we'll add a multi-key trust store (verify against any of N public keys, sign with the latest).

## Regenerating (one-time setup only)

```bash
openssl genpkey -algorithm Ed25519 -out license_private.pem
openssl pkey -in license_private.pem -pubout -out license_pub.pem
```

The private key goes to `~/.safetylens/license_private.pem` (chmod 600). The public key goes here.
