# Jetson four-frame TensorRT microbatch experiment (2026-07-12)

> **2026-07-13 update:** the 14 ms partial-group smoothing follow-up promotes
> 21 cameras at 4 FPS. See
> `jetson-partial-group-smoothing-2026-07-13.md` for the current boundary.

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

This experiment tests whether the proven two-frame cross-camera microbatch can
be widened to four frames without changing the Small-model accuracy contract.
It does not use a Nano model.

## Decision

Promote the opt-in batch-4 capacity profile on the tested Orin NX. It raises
the measured conditional inference tier from 18 to 20 camera-equivalents at
4 FPS with 11.1% PPE specialist duty. Twenty completed the 60-second cold gate
with zero drops, failures, route fallbacks, or timeout fallbacks. Twenty-one is
rejected because its cold maximum crossed the 250 ms camera period; twenty-two
dropped primary requests.

Keep the batch-2 profile as the immediate rollback and as the lower-latency
choice for deployments with fewer than four simultaneously active cameras.

## Candidate engines

Both engines use fixed 640-pixel input and batch size four.

| Role | Model | Precision | Engine SHA-256 | TensorRT context allocation |
| --- | --- | --- | --- | ---: |
| Primary | YOLO26 Small | INT8 | `77975ac0f311cbe5f873f42317c7127b61387b1d2fec1e9f160324691b1404ba` | 51 MiB |
| PPE specialist | YOLOE-26S | FP16 | `89f97d6731d8b323ae7e65554a5be608ec89673a8f0fa871230e129528a5b1a7` | 75 MiB |

The primary engine reused the same 347-readable-image calibration corpus and
calibration YAML digest as the promoted batch-2 INT8 engine. The PPE engine
keeps the fixed helmet prompt set and semantic grouping used in production.

## Equal-work throughput

Each sample represents four frames. The batch-2 baseline executes two engine
calls; the candidate executes one batch-4 call. Each result below used ten
warmups and 200 timed samples in a fresh isolated GPU container.

| Path | Batch-2 throughput | Batch-4 throughput | Gain | Batch-2 median | Batch-4 median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary, 13 phone/office frames | 91.349 FPS | 104.643 FPS | 14.6% | 44.513 ms | 39.097 ms |
| PPE, 18 validation frames | 64.058 FPS | 71.079 FPS | 11.0% | 62.459 ms | 56.296 ms |

Primary output was exact across all 77 detections. For PPE, all 13 batch-2
detections matched the batch-4 output with minimum IoU 0.9891 and maximum
confidence drift 0.0042. Batch-4 added one overlapping class-2 proposal at
0.2947 confidence on `factory-ppe-gate.jpg`; the production semantic prompt
deduplicator removes overlapping helmet synonyms before alert evaluation.

## Runtime implementation

The candidate is opt-in and leaves the existing batch-2 deployment contract
intact:

- separate manifest-validated batch-4 primary and PPE engine variables;
- four-frame raw primary and specialist routes;
- bounded four-camera grouping before one remote admission slot;
- concurrent primary/PPE execution for specialist groups;
- configurable phase grouping of four cameras;
- `deploy/jetson-batch2.env.example` remains the rollback profile.

Local regression status after the live gate: 293 focused tests passed,
including four-frame runtime, transport, route, scaling, specialist, compose,
health, scheduler, capture, and video-processing coverage.

## Runtime contract smoke

An isolated candidate model server reported both batch-2 and batch-4 primary
and PPE TensorRT engines configured, warmed, and fixed at 640 pixels. A
four-camera, 10-second edge smoke produced:

- 120/120 primary requests in 30 batch-4 calls;
- 40/40 PPE requests in 10 batch-4 calls;
- zero route or timeout fallbacks;
- zero overloads or failures;
- 4.0 FPS minimum, 81.807 ms p95, and 94.835 ms maximum.

## Exact edge-load boundary

The load gate exercised candidate `predict_record_batches()` with 4 FPS per
camera, 640-pixel YOLO26 Small INT8 primary inference, 11.1% YOLOE-26S PPE
duty, four-camera phases, 10 ms primary/PPE rendezvous windows, four admission
slots, and a 125 ms bounded wait.

| Cameras | Duration | Completed | Drops / failures | Primary batch-4 calls | PPE batch-4 calls | Timeout fallbacks | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 30 s | 2,400 / 2,400 | 0 / 0 | 535 | 65 | 0 | 163.516 ms | 209.334 ms | provisional |
| 21 | 30 s | 2,520 / 2,520 | 0 / 0 | 535 | 65 | 120 | 168.545 ms | 233.584 ms | provisional with remainder fallback |
| 22 | 30 s | 2,636 / 2,640 | 4 / 0 | 534 | 65 | 240 | 196.818 ms | 271.202 ms | reject |
| 21 | 60 s cold | 5,040 / 5,040 | 0 / 0 | 1,070 | 130 | 240 | 179.778 ms | 258.433 ms | reject: stale tail |
| 20 | 60 s cold | 4,800 / 4,800 | 0 / 0 | 1,070 | 130 | 0 | 141.233 ms | 171.341 ms | promote |

The odd-camera remainder explains the batch-4 timeout fallbacks at 21. Request
completion alone is not sufficient: the 60-second maximum exceeded the next
250 ms inference period. Twenty is the highest clean tier; twenty-two is the
measured overload boundary.

## Live deployment and rollback

The candidate edge and model-server images were deployed with the previous
batch-2 containers preserved intact under rollback names. The model server
reported all four fixed-batch runtimes configured and warmed. Cam2 returned
fresh on `gstreamer_nvdec` with zero inference overloads or failures. Cam1
remained in its pre-existing source outage.

A 60-second live soak on cam2 produced 168 successful inferences, zero new
overloads, and zero failures. Only one camera was frame-fresh, so live traffic
correctly used the singleton timeout fallback rather than pretending to form a
four-frame group. That adds the 10 ms rendezvous wait and means batch-2 remains
the more latency-efficient profile for one-to-three active cameras.

Rollback was exercised, not assumed:

1. stop and preserve the candidate containers;
2. restore the original batch-2 containers;
3. verify model health and fresh cam2 NVDEC inference with zero failures;
4. switch forward to the already-tested batch-4 containers;
5. verify the same health and camera invariants again.

The final live state uses the batch-4 images. The stopped batch-2 containers
remain available for immediate rollback, and the model watchdog timer is
active.

The live swap also exposed an operational engineering gap: deployment still
relies on ad hoc `docker run` container reconstruction instead of one
versioned, idempotent promote/rollback command. An initial operator-side mount
cloning command failed on shell quoting and a double-triggered cleanup removed
the active container names. Model data, database volumes, camera configuration,
and images were unaffected; the known-good batch-2 containers were recreated
and health-checked before the corrected swap. Container promotion automation
should be codified before the next runtime profile change.

Raw benchmark evidence is stored on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/batch4/`
