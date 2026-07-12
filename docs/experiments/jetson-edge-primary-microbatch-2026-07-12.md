# Jetson edge-side primary microbatch experiment — 2026-07-12

## Decision

Keep the implementation as an opt-in candidate and retain the known-good
production containers until the fifteen-camera result is repeated. Pairing two
compatible primary-only frames on the edge *before* remote admission moves the
measured conditional Small-model inference boundary from fourteen to fifteen
camera-equivalents at 4 FPS. Sixteen remains beyond the zero-drop boundary.

This is an inference-capacity result, not a claim that fifteen simultaneous
RTSP decoders have been validated. The two office cameras were restored after
the isolated benchmark and both camera workers restarted on the known-good
production images.

## Architecture under test

The candidate path is:

`paired camera phases -> bounded 6 ms edge rendezvous -> one admission slot -> one raw two-frame request -> YOLO26 Small INT8 batch-2`

Only singleton `coco_primary` requests at 640px are eligible. Frames with PPE
or another specialist keep the existing grouped-model route. Different
confidence, device, resolution, or class-filter settings cannot pair. An
unmatched request falls through to the existing single-frame transport, and a
404 from an older model server disables further rendezvous waits for that
server URL. Both paired callers receive the same admission, HTTP, validation,
or inference failure.

The feature is disabled by default. It requires all of the following to be set
for a validated device:

- a manifest-validated fixed batch-2 Small TensorRT engine;
- a non-zero `SAFETYLENS_REMOTE_PRIMARY_BATCH_WAIT_SECONDS`;
- paired camera phases through `SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE=2`;
- device-specific admission settings established by the load gate.

## Accuracy and engine gate

The batch-2 engine uses the same YOLO26 Small source and 347-frame INT8
calibration corpus as the deployed batch-1 engine. The prior isolated engine
gate returned exactly matching class IDs, confidences, and boxes across four
control frames. Batch-2 delivered 99.784 frames/second versus 81.507 for two
sequential INT8 batch-1 calls, a 22.4% aggregate throughput gain.

The new end-to-end run preserved every scheduled specialist evaluation at the
fifteen-camera passing tier: 195 of 195. The sixteen-camera failure lost four
scheduled evaluations, so it is not a supported tier.

## Exact edge-load result

The harness exercised the real edge `predict_record_batches()` path with raw
transport, 640px YOLO26 Small INT8 primary inference, 11.1% Small PPE duty,
phone/PPE overlap avoidance, and 4 FPS per camera. Each run lasted 30 seconds.
The candidate model server kept both batch-1 and batch-2 engines resident, so
the same-window disabled run is the fair comparison rather than an older warm
baseline from a different server process.

| Cameras / pairing / slots / admission wait | Completed | Drops | Specialist calls | Primary pairs | Timeouts | Median | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 / disabled / 3 / 75 ms | 1662 / 1680 | 18 | 176 | 0 | 0 | 16.971 ms | 102.409 ms | 296.909 ms |
| 14 / 6 ms / 3 / 75 ms | 1678 / 1680 | 2 | 180 | 749 | 0 | 25.921 ms | 107.073 ms | 159.123 ms |
| 14 / 6 ms / 4 / 75 ms | 1679 / 1680 | 1 | 181 | 749 | 0 | 25.826 ms | 108.394 ms | 180.869 ms |
| 14 / 6 ms / 4 / 100 ms | 1680 / 1680 | 0 | 182 | 749 | 0 | 25.842 ms | 109.277 ms | 182.230 ms |
| 15 / 6 ms / 4 / 100 ms | 1800 / 1800 | 0 | 195 | 749 | 107 | 26.064 ms | 134.277 ms | 201.362 ms |
| 16 / 6 ms / 4 / 100 ms | 1916 / 1920 | 4 | 204 | 856 | 0 | 26.068 ms | 141.926 ms | 226.679 ms |

All 749 eligible pairs at fourteen and fifteen cameras executed without a
pairing timeout. Fifteen is odd, so its remaining camera correctly exercised
107 single-frame timeout fallbacks. The passing maximum latency of 201.362 ms
remained inside the 250 ms frame period. Sixteen crossed the zero-drop gate
despite all primary-only frames pairing.

## Bottlenecks exposed

1. Specialist bursts now define the boundary. Primary pairing removes most
   primary contention, but specialist frames deliberately bypass this batch
   engine and can still synchronize across paired cameras.
2. Four bounded admission slots and a 100 ms wait are required for the passing
   tier on this Jetson. Three slots or the former 75 ms bound still shed one or
   more jobs in the same-window candidate runs.
3. The batch-2 engine is currently an external device artifact, not a versioned
   deployment asset. Enabling the edge rendezvous without the matching engine
   would merely serialize two batch-1 calls on the server and should not be
   promoted.
4. Candidate image assembly exposed base-image skew: an older server tag did
   not contain the current manifest validator's `expected_batch` support. The
   corrected candidate included the current `tensorrt_engine.py`; immutable
   build provenance should replace ad-hoc candidate tag inheritance.

## Promotion gate

Repeat the fifteen-camera 1800/1800 run from a clean candidate startup, then
run the two live office cameras on the opt-in settings and require fresh frames,
zero inference overloads/failures, correct alert routing, and successful
rolling fallback with the batch route disabled. Until that completes, the
checked-in defaults remain off and production stays on the certified
fourteen-camera configuration.
