# Jetson container promotion and rollback

Use `scripts/jetson_container_swap.py` instead of reconstructing production
containers with ad hoc shell commands. The tool clones the inspected runtime
configuration with subprocess argument arrays, so environment values are not
interpolated by a shell or printed in its result.

It preserves environment, mounts, devices, ports, restart policy, network,
resource limits, logging, entrypoint, and command. Promotion first creates and
removes an exact staging container before stopping the active service. If the
candidate cannot start or its HTTP health endpoint does not return 200, the
tool removes the candidate, restores the old name, starts the rollback, and
checks it again.

## Promote

The rollback name must not exist. Always run the read-only preflight first:

```bash
python3 scripts/jetson_container_swap.py \
  --active rakshak-edge \
  --image rakshak-lens-edge:candidate-adaptive-microbatch \
  --rollback rakshak-edge-pre-adaptive \
  --health-url http://127.0.0.1:8000/api/health \
  --set-env SAFETYLENS_REMOTE_BATCH2_EARLY_FLUSH_SECONDS=0.006 \
  --dry-run
```

Remove `--dry-run` only after the reported mount and device counts match the
active container. Use a root-readable `--env-file` for a larger runtime
profile. The output lists overridden variable names but never their values.

## Restore a preserved container

Rollback retains the displaced candidate, enabling a second verified swap
back to it:

```bash
python3 scripts/jetson_container_swap.py \
  --active rakshak-edge \
  --rollback rakshak-edge-pre-adaptive \
  --restore \
  --displaced rakshak-edge-adaptive-tested \
  --health-url http://127.0.0.1:8000/api/health
```

The HTTP check proves process readiness. After either transition, separately
verify the SafetyLens health payload for fresh expected cameras, the intended
capture backend, and zero new inference overloads or failures. A pre-existing
camera source outage may legitimately keep overall health degraded even when
the replacement container is sound.

## Validated live transition

On 2026-07-12 the tool promoted the adaptive batch candidate, verified the
edge endpoint, restored the preserved batch-4 container, and restored the
adaptive candidate again. Every existing environment value was retained; the
adaptive early-flush setting was the only added variable. Mount and device
sets, network, port, restart policy, and resource configuration matched the
original container. The final active state is the adaptive image, with the
batch-4 container stopped under its rollback name. A separate
failure-injection promotion used an unreachable health endpoint; the tool
removed the failed candidate, restored the original active container and
environment, and returned the real health endpoint to HTTP 200.
