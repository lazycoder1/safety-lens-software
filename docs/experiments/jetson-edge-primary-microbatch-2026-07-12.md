# Jetson edge-side primary microbatch experiment — 2026-07-12

## Decision

Promote the opt-in profile on the tested Orin NX while retaining the previous
containers for rollback. Pairing two compatible primary-only frames on the
edge *before* remote admission moves the measured conditional Small-model
inference boundary from fourteen to fifteen camera-equivalents at 4 FPS.
Sixteen remains beyond the zero-drop boundary.

This is an inference-capacity result, not a claim that fifteen simultaneous
RTSP decoders have been validated. The two office cameras are running the
candidate with the previous production images preserved as stopped rollback
containers. The reusable non-secret settings are recorded in
`deploy/jetson-batch2.env.example`; generic defaults remain disabled because
the batch-2 engine is an external, device-specific artifact.

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
phone/PPE overlap avoidance, and 4 FPS per camera. Runs lasted 30 seconds unless
the table identifies the 60-second promotion gate. The candidate model server
kept both batch-1 and batch-2 engines resident, so
the same-window disabled run is the fair comparison rather than an older warm
baseline from a different server process.

| Cameras / pairing / slots / admission wait | Completed | Drops | Specialist calls | Primary pairs | Timeouts | Median | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 / disabled / 3 / 75 ms | 1662 / 1680 | 18 | 176 | 0 | 0 | 16.971 ms | 102.409 ms | 296.909 ms |
| 14 / 6 ms / 3 / 75 ms | 1678 / 1680 | 2 | 180 | 749 | 0 | 25.921 ms | 107.073 ms | 159.123 ms |
| 14 / 6 ms / 4 / 75 ms | 1679 / 1680 | 1 | 181 | 749 | 0 | 25.826 ms | 108.394 ms | 180.869 ms |
| 14 / 6 ms / 4 / 100 ms | 1680 / 1680 | 0 | 182 | 749 | 0 | 25.842 ms | 109.277 ms | 182.230 ms |
| 15 / 6 ms / 4 / 100 ms | 1800 / 1800 | 0 | 195 | 749 | 107 | 26.064 ms | 134.277 ms | 201.362 ms |
| 15 / 6 ms / 4 / 100 ms, clean repeat | 1798 / 1800 | 2 | 193 | 749 | 107 | 25.950 ms | 137.744 ms | 211.313 ms |
| 15 / 6 ms / 4 / 125 ms, 60 seconds | 3600 / 3600 | 0 | 390 | 1498 | 214 | 25.871 ms | 132.703 ms | 225.699 ms |
| 16 / 6 ms / 4 / 100 ms | 1916 / 1920 | 4 | 204 | 856 | 0 | 26.068 ms | 141.926 ms | 226.679 ms |

The first 100 ms fifteen-camera run passed, but its clean-start repeat dropped
two specialist frames. It was therefore not promoted. Raising only the bounded
admission wait to 125 ms completed a longer 3,600-request gate with all 390
specialist calls preserved. Fifteen is odd, so its remaining camera correctly
exercised 214 single-frame timeout fallbacks. The 225.699 ms maximum remained
inside the 250 ms frame period. Sixteen crossed the zero-drop gate despite all
primary-only frames pairing.

## Bottlenecks exposed

1. Specialist bursts now define the boundary. Primary pairing removes most
   primary contention, but specialist frames deliberately bypass this batch
   engine and can still synchronize across paired cameras.
2. Four bounded admission slots and a 125 ms wait are required for the repeated
   passing tier on this Jetson. The 100 ms setting produced one clean run but
   shed two specialist jobs after a clean restart.
3. The batch-2 engine is currently an external device artifact, not a versioned
   deployment asset. Enabling the edge rendezvous without the matching engine
   would merely serialize two batch-1 calls on the server and should not be
   promoted.
4. Candidate image assembly exposed base-image skew: an older server tag did
   not contain the current manifest validator's `expected_batch` support. The
   corrected candidate included the current `tensorrt_engine.py`; immutable
   build provenance should replace ad-hoc candidate tag inheritance.

## Live-camera promotion

The exact pushed revision was rolled out with the old edge and model-server
containers retained for atomic rollback. During the live soak:

- both `cam1` and `cam2` stayed `online` and `running`;
- 1,050 inference cycles completed with zero overload drops and zero failures;
- recent detections included person, motorcycle, car, truck, bicycle, helmet,
  bowl, and cup;
- 15 real two-frame batch requests executed without a route or inference error;
- 1,013 eligible requests used the bounded singleton fallback because the
  static and active cameras ran at different motion-adaptive cadences;
- the alert persistence worker remained accepting and failure-free, and the
  delivery outbox had zero due work.

An ordered model-server then edge restart reloaded the batch-2 engine, restored
both cameras to `online`/`running`, executed seven new primary pairs in the
first 30 seconds, and retained zero overloads and failures. The validated
profile is also installed on the device as
`/opt/rakshak-lens/jetson-batch2-850650b.env`.

No new qualifying violation occurred during the soak, so it did not create a
new semantic alert. The unchanged alert persistence and delivery path remained
healthy; alert-positive behavior continues to rely on the existing rule tests
and labelled replay gates rather than manufacturing a live external alert.

The health endpoint now exposes non-sensitive primary-batch counters under
`inferenceTransport.primaryFrameBatch`, making pairing, fallback, overload, and
rolling-route behavior operationally visible.
