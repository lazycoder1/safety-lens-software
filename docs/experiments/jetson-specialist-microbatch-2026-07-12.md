# Jetson conditional-specialist microbatch experiment — 2026-07-12

## Decision

Promote edge-side pairing for compatible `coco_primary + ppe_specialist`
requests on the tested Orin NX. Together with the existing Small-only primary
filter and primary-frame batching, this moves the measured conditional
inference tier from fifteen to sixteen camera-equivalents at 4 FPS.

Seventeen is rejected because its 254.283 ms maximum request latency exceeds
the 250 ms frame period. Eighteen is rejected because it dropped eight
requests. This is an inference-capacity result: the site has separately
validated four simultaneous real RTSP/NVDEC streams, not sixteen decoders.

The production path remains Small-only. Nano is not used.

## Pipeline under test

The active cascade is:

1. decode and sample each camera at its configured inference cadence;
2. run YOLO26 Small INT8 at 640px as the cheap full-frame COCO primary;
3. reuse the fresh primary result for person, vehicle, zone, and association
   eligibility checks;
4. run YOLOE-26S PPE only when the camera rules and primary context can produce
   an actionable alert;
5. pair two compatible cross-camera primary-plus-PPE requests before remote
   admission;
6. execute one fixed-batch-2 primary inference followed by one fixed-prompt
   batch-2 PPE inference;
7. return per-camera results to the unchanged rule, cooldown, persistence, and
   delivery path.

Only an exact two-model request is eligible. Resolution, thresholds, device,
primary class filter, PPE prompt order, and semantic groups are part of the
pairing key. An unmatched request times out after a bounded 6 ms rendezvous and
uses the existing grouped single-frame route. Older servers, unavailable
batch-2 engines, and prompt mismatches also fall back without dropping the
frame.

## PPE batch-2 engine gate

The candidate is a fixed-shape FP16 YOLOE-26S segmentation engine at 640px and
batch two. Its prompts are fixed to `motorcycle helmet`, `rider helmet`, and
`helmet`, all mapped to the `rider_helmet_required` semantic group.

- engine size: 22 MiB;
- build time: 584.703 seconds;
- low-memory TensorRT workspace: 256 MiB;
- TensorRT allocator peak GPU memory: 279 MiB;
- final assigned activation memory: 39,014,912 bytes;
- engine SHA-256:
  `12f3376d54c882e3fa1a048e05ed7e77ab975e8ae3f321718ab8aae4384b27ee`.

Each engine was measured in a fresh isolated model-server container over 200
timed frame pairs.

| Shape | Aggregate FPS | Median pair | p95 pair | Maximum pair |
| --- | ---: | ---: | ---: | ---: |
| Two sequential batch-1 PPE calls | 55.757 | 35.068 ms | 39.287 ms | 42.111 ms |
| One batch-2 PPE call | 63.947 | 30.376 ms | 34.444 ms | 34.559 ms |

Batch two improved aggregate PPE throughput by 14.7%.

All 18 PPE validation images were also compared at confidence 0.20. Both
engines produced 13 detections with identical class IDs. The maximum confidence
delta was 0.0044, the maximum box-coordinate delta was 0.2 pixels, and the
minimum paired box IoU was 0.989071. This is acceptable FP16 tactic drift and
not the accuracy collapse previously observed with the rejected INT8 PPE
engine.

## Exact edge-load capacity

The harness exercised the real edge `predict_record_batches()` path at 4 FPS
per camera with YOLO26 Small INT8 primary inference, 11.1% conservative PPE
duty, four bounded admission slots, a 125 ms admission wait, a 6 ms primary
rendezvous, and a 6 ms conditional-specialist rendezvous.

| Cameras | Duration | Completed | Drops / failures | Specialists | Primary pairs | Specialist pairs | Median | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 30 s | 1,920 / 1,920 | 0 / 0 | 208 / 208 | 856 | 104 | 26.184 ms | 120.876 ms | 209.180 ms | pass |
| 17 | 30 s | 2,040 / 2,040 | 0 / 0 | 221 / 221 | 856 | 104 | 26.057 ms | 136.078 ms | 254.283 ms | reject: stale boundary |
| 18 | 30 s | 2,152 / 2,160 | 8 / 0 | 234 completed | 856 | 104 | 26.184 ms | 179.417 ms | 299.524 ms | reject: drops |
| 16 | 60 s clean repeat | 3,840 / 3,840 | 0 / 0 | 416 / 416 | 1,712 | 208 | 26.057 ms | 118.532 ms | 210.856 ms | promote |

The 60-second clean-start repeat is the promotion gate. It preserved every
scheduled specialist evaluation and completed every camera at exactly 4 FPS.

## Operational behavior

The feature is opt-in because both batch-2 TensorRT engines are device-specific
external artifacts. Generic defaults remain disabled. The reusable non-secret
profile is recorded in `deploy/jetson-batch2.env.example`.

The health endpoint exposes specialist pairing and fallback counters under
`inferenceTransport.specialistFrameBatch`. On the live two-camera office soak,
both cameras remained online with zero inference drops and failures, the alert
pipeline remained accepting and persistence remained healthy. Only cam2 had a
rider-PPE plan, so its four specialist calls correctly used the bounded
single-frame fallback rather than manufacturing a cross-camera pair. Recent
live detections included person, motorcycle, car, and truck.

## Remaining boundaries

- Sixteen is valid for the measured 11.1% conditional PPE workload, not for an
  always-on specialist on all cameras.
- Real RTSP/NVDEC ingest is validated only to four simultaneous streams at this
  site. A larger NVR source or RTSP simulator is required for an end-to-end
  sixteen-stream certification.
- Mobile-phone accuracy remains 4/6 labelled positives with zero actionable
  negatives on the deployed 640px Small INT8 path. RT-DETR-L and RF-DETR Small
  did not provide a deployable accuracy/latency tradeoff on this Jetson.
- Tracking was not promoted as a compute filter. The tested ByteTrack path
  added tail latency and did not safely bridge sparse primary detections.
- The engine artifacts need immutable build provenance and distribution; the
  repository currently versions the runtime and configuration contract, not
  the hardware-specific TensorRT binaries.
