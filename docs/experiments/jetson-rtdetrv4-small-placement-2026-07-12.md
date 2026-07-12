# Jetson RT-DETRv4 Small placement experiment — 2026-07-12

> **2026-07-13 capacity note:** the no-RT-DETR batch-4 tier is 21 cameras at
> 4 FPS after partial-group smoothing. Phase-aware substitution repeatably
> sustained 20 cameras at an effective 4 FPS plus one device-wide RT-DETRv4-S
> FPS, with every PPE pass preserved and maximum primary latency below 235 ms.
> The 21-camera probes exceeded the 250 ms freshness limit. The substitution
> topology is benchmark-proven but is not yet integrated into live alerts.

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

Do not replace the YOLO26 Small INT8 primary with RT-DETRv4 Small. The
RT-DETRv4 batch-2 engine sustained 43.792 end-to-end FPS versus 91.349 FPS for
the then-current batch-2 primary and 104.643 FPS for the subsequently promoted
batch-4 primary. Its freshness-safe engine-only boundary was ten cameras at
4 FPS; eleven cameras produced sustained staleness before adding PPE,
transport, or alert processing.

RT-DETRv4-S is useful only as a conditional phone-recall specialist after the
current primary, tracker/context, and zone rules. On the labelled Jetson set it
improved actionable phone recall from 4/6 to 5/6 while preserving zero
actionable hits across four negative controls. A separate TensorRT context
proved these conservative mixed-load tiers:

- 16 cameras at 4 FPS plus one aggregate RT-DETRv4 specialist FPS;
- 15 cameras at 4 FPS plus two aggregate RT-DETRv4 specialist FPS.

The current no-RT-DETR pipeline supports 21 camera-equivalents at 4 FPS. An
earlier additive follow-up found an 18-camera zero-drop throughput boundary but
unacceptable tail jitter. Replacing selected scheduled YOLO frame slots instead
of adding work raised the repeatably freshness-safe experimental boundary to
20 cameras at effective 4 FPS plus one device-wide RT-DETRv4 specialist FPS.
RT-DETRv4 was not deployed because its results are not yet merged into the live
tracker and alert path.

## Question

Test the smallest official RT-DETRv4 model rather than extrapolating from the
previous RT-DETR Large result. Decide whether it should replace YOLO26 Small as
the primary COCO detector or run conditionally downstream.

The official RT-DETRv4 release provides Small, Medium, Large, and X variants.
There is no official Nano checkpoint, so Small is the lightest supported test.

## Candidate contract

- checkpoint: `rtv4_hgnetv2_s_coco.pth`;
- upstream revision: `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`;
- checkpoint SHA-256:
  `238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee`;
- deploy parameters: 10,434,867;
- input: RGB float tensor at 640 by 640 plus original image size;
- output: 300 COCO class, box, and confidence queries;
- COCO class IDs are contiguous 0 through 79 and match the current alert-class
  contract;
- official distillation is training-only and adds no inference-time component.

The checkpoint loaded without missing or unexpected keys. The fixed batch-1
opset-16 ONNX graph passed `onnx.checker`; PyTorch versus ONNX produced exact
labels, maximum box drift of 0.00879 pixels, and maximum score drift of
0.000001326 on a random parity input.

Fixed batch-2 and batch-4 graphs were also exported for cross-camera batching.
Across eight local office frames, both graphs preserved every person at
confidence 0.30 and every phone at confidence 0.15 relative to batch-1. Greedy
class matching produced minimum IoU 0.999988 and maximum confidence drift
0.0000179.

## Local alert-quality pre-screen

Four visually checked phone-use frames and four negative office/factory frames
were evaluated before spending Jetson build time. The evaluation used the
production 0.30 person threshold, 0.15 phone threshold, and exact
phone-to-person geometry.

At confidence 0.15, RT-DETRv4-S produced actionable phone detections on two of
four positive frames and one of four negative frames. Raising only the phone
threshold to 0.18 retained the same two positives and removed that local
negative hit. It still missed the distant phone-use frame and produced a phone
box outside the production person association on the fourth positive.

Class-aware NMS did not remove the negative phone boxes; they were distinct
queries rather than duplicate boxes. NMS reduced some low-confidence query
duplication, but does not solve the alert-quality issue by itself. The current
YOLO26 Small checkpoint produced no actionable phone detections on these four
local positive frames and none on the four negatives at 0.15. This small local
set is a pre-screen, not a replacement for the existing thirteen-frame Jetson
corpus.

## Current production baseline

At the time of the first RT-DETRv4-S comparison, the proven tier was 18
camera-equivalents at 4 AI detection FPS per camera, 640-pixel YOLO26 Small
INT8 primary inference, and 11.1% YOLOE-26S PPE duty. The subsequently promoted
batch-4 route first raised that conditional inference tier to 20 camera
equivalents. Partial-group smoothing then raised the current tier to 21, or 84
scheduled primary frame-inferences per second plus about 9.3 PPE specialist
frame-inferences per second. These are inference-compute tiers, not claims that
the same number of unique production RTSP sources has been validated end to
end.

## Placement gate

RT-DETRv4-S must not replace the primary merely because it improves one phone
frame. Primary promotion requires person, vehicle/motorcycle, animal, phone,
zone-routing, latency, and false-alert parity on the Jetson corpus. It also must
preserve the current 18 by 4 FPS tier under the mixed specialist load.

RT-DETRv4-S cannot replace the PPE specialist because COCO has no helmet, vest,
apron, or harness class contract. If it improves phone recall without primary
capacity parity, the viable placement is a conditional phone specialist after
the current primary person/zone gate, at a bounded cadence rather than on every
camera frame.

## TensorRT build result

The fixed batch-1 and batch-2 FP16 engines built successfully with a 256 MiB
workspace cap. This is materially better than the previous Large model, whose
two TensorRT export/build attempts were OOM-killed.

| Batch | Build time | Engine size | Engine SHA-256 | Runtime activation memory |
| ---: | ---: | ---: | --- | ---: |
| 1 | 1,141 s | 25 MiB | `283342ff8bf4f6a5d817485060a2a06e50ae547765f2fc2edf8c84d8c6a96167` | 23,102,976 bytes |
| 2 | 1,211 s | 25 MiB | `073f743178fabb82f5f0147b14665cc34ff62a7ada4dfb75b9c7469a5f2b19b5` | 46,203,904 bytes |

TensorRT emitted the same FP16 conversion warnings for both engines: one FP32
infinity, 140 subnormal values, ten values below the smallest FP16 subnormal,
and two finite FP32 values that overflow FP16. Output accuracy was therefore
gated rather than assumed. The shared timing cache grew from 890 KiB after
batch-1 to 1.8 MiB after batch-2, but fixed batch shapes still required about
twenty minutes of profiling each.

Batch-4 was not built. Batch-2 improved end-to-end throughput by only 14.0%
over batch-1 and remained less than half as fast as the current primary. A
further long production-offline build could not plausibly recover the gap or
create the PPE headroom required for primary promotion.

## Isolated Jetson throughput

Each result used ten warmups and 200 timed samples over all thirteen existing
office/phone frames.

| Engine | Measurement | Throughput | Median | p95 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: |
| RT-DETRv4-S batch-1 FP16 | model-only | 58.028 FPS | 17.165 ms | 17.215 ms | 17.885 ms |
| RT-DETRv4-S batch-1 FP16 | end-to-end | 38.424 FPS | 25.998 ms | 26.438 ms | 27.672 ms |
| RT-DETRv4-S batch-2 FP16 | model-only | 66.897 FPS | 29.827 ms / 2 frames | 29.900 ms | 30.086 ms |
| RT-DETRv4-S batch-2 FP16 | end-to-end | 43.792 FPS | 45.602 ms / 2 frames | 46.188 ms | 53.179 ms |
| YOLO26 Small batch-2 INT8 | deployed end-to-end comparison | 91.349 FPS | 44.513 ms / 4 frames | — | — |

Even the RT-DETRv4 model-only throughput was below the current YOLO end-to-end
throughput. RT-DETRv4-S therefore cannot preserve the current 72 primary
frame-inferences per second plus PPE duty.

## Labelled Jetson phone accuracy

At the production 0.30 person confidence, 0.15 phone confidence, and exact
phone-to-person geometry:

| Model | Actionable positive images | Actionable negative images |
| --- | ---: | ---: |
| YOLO26 Small INT8 | 4/6 | 0/4 |
| RT-DETRv4-S FP16 batch-1 | 5/6 | 0/4 |
| RT-DETRv4-S FP16 batch-2 | 5/6 | 0/4 |

RT-DETRv4-S recovered `pos-dark.jpg` and missed `pos-worker.jpg`. It emitted
high-confidence raw phone boxes on all four negative controls, including scores
above 0.74 on `neg-arm-table.jpg` and `neg-laptop.jpg`; production person
association suppressed them. It must not be used as an unfiltered phone-alert
source.

Batch-2 preserved the actionable result for every labelled image. Small score
and person-query differences remained after FP16 conversion, so the batch
engine should continue to be evaluated through alert outcomes rather than raw
query equality.

## Primary cadence boundary

The batch-2 engine was paced as an always-on primary over the thirteen-frame
corpus. The production-aligned stale budget was one 250 ms camera inference
period, rather than the much shorter interval between paired camera groups.

| Aggregate target | Camera equivalent at 4 FPS | Duration | Stale groups | Maximum start lateness | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 36 FPS | 9 | 60 s | 0 / 1,080 | 4.196 ms | pass |
| 40 FPS | 10 | 60 s | 0 / 1,200 | 61.271 ms | pass |
| 44 FPS | 11 | 30 s | 370 / 660 | 385.558 ms | reject |

Ten cameras is an engine-only ceiling, not a promoted mixed production tier.
It excludes PPE, model-server transport, and alert evaluation, all of which
need additional headroom.

## Conditional specialist capacity

The exact edge load used YOLO26 Small at 4 FPS per camera, 11.1% YOLOE-26S PPE
duty, paired camera phases, batch-2 primary and PPE routes, four admission
slots, and a 125 ms bounded wait. RT-DETRv4 ran in a separate paced batch-2
TensorRT context. This is a conservative integration shape because a future
same-process scheduler may avoid some context-switch and burst interference.

| Cameras | RT-DETR aggregate FPS | Duration | Primary successes | Drops / failures | Minimum camera FPS | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 18 | 0 | 30 s | 2,160 / 2,160 | 0 / 0 | 4.000 | 132.999 ms | 190.013 ms | baseline pass |
| 18 | 2 | 30 s | 2,138 / 2,160 | 22 / 0 | 3.867 | 164.811 ms | 477.616 ms | reject |
| 17 | 2 | 30 s | 2,040 / 2,040 | 0 / 0 | 4.000 | 155.431 ms | 224.696 ms | provisional |
| 17 | 2 | 60 s | 4,070 / 4,080 | 8 / 2 | 3.967 | 154.462 ms | 322.298 ms | reject |
| 16 | 2 | 60 s | 3,836 / 3,840 | 4 / 0 | 3.983 | 117.263 ms | 288.483 ms | reject |
| 17 | 1 | 60 s | 4,074 / 4,080 | 6 / 0 | 3.967 | 149.899 ms | 283.983 ms | reject |
| 16 | 1 | 60 s | 3,840 / 3,840 | 0 / 0 | 4.000 | 115.086 ms | 239.428 ms | pass |
| 15 | 2 | 60 s | 3,600 / 3,600 | 0 / 0 | 4.000 | 124.619 ms | 218.398 ms | pass |

Both passing RT-DETR workloads achieved their target FPS with zero stale
specialist groups. The results support a device-wide specialist budget tied to
currently actionable contexts, not one RT-DETR invocation per camera.

## Initial batch-4 coexistence boundary

After the four-frame primary and PPE routes raised the no-RT-DETR tier to 20
cameras, the 1 FPS conditional specialist gate was repeated against the exact
current edge transport. The test used 640-pixel YOLO26 Small INT8 primary at
4 FPS per camera, 11.1% YOLOE-26S PPE duty, four-camera phases, adaptive
batch-2/batch-4 routing, four admission slots, and a 125 ms bounded wait.
RT-DETRv4-S ran in a separate prewarmed TensorRT context for the full camera
load interval.

| Cameras | RT-DETR aggregate FPS | Duration | Primary successes | Drops / failures | Minimum camera FPS | Primary p95 | RT-DETR stale groups | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 19 | 1 | 60 s | 4,559 / 4,560 | 1 / 0 | 3.983 | 181.751 ms | 0 | reject |
| 18 | 1 | 60 s | 4,320 / 4,320 | 0 / 0 | 4.000 | 147.968 ms | 0 | pass |

The passing RT-DETR process achieved 1.013 FPS, with 51.094 ms median and
69.980 ms p95 end-to-end group latency. The result establishes 18 cameras as
the measured throughput tier when the deployment spends one aggregate
RT-DETRv4-S frame per second on phone-recall escalation. It does not authorize
one RT-DETR pass per camera per second; that would request 18 aggregate
specialist FPS and overload this device.

## Partial-group profile re-gate — 2026-07-13

The conditional load was repeated after the primary/PPE partial-group wait was
raised to 14 ms and the no-RT-DETR tier reached 21 cameras. The workload kept
the current 640-pixel YOLO26 Small INT8 primary at 4 FPS per camera, 11.1%
YOLOE-26S PPE duty, four-camera phases, batch-4 routing, four admission slots,
and a 125 ms admission bound. RT-DETRv4-S again ran at one aggregate FPS in a
separate prewarmed TensorRT context.

The freshness gate requires every scheduled request to complete, every camera
to sustain 4 FPS, and maximum primary latency to remain below the 250 ms camera
period. Runs that complete every request but exceed 250 ms are useful
throughput evidence, not production-safe freshness evidence.

| Cameras | Duration | Primary successes | Drops / failures | Minimum camera FPS | Primary p95 | Primary maximum | RT-DETR achieved | RT-DETR stale groups | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 19, no RT control | 30 s | 2,280 / 2,280 | 0 / 0 | 4.000 | 142.897 ms | 226.874 ms | — | — | pass control |
| 19 + RT | 30 s | 2,272 / 2,280 | 8 / 0 | 3.933 | 197.316 ms | 295.092 ms | 1.022 FPS | 0 | reject |
| 18 + RT | 60 s | 4,320 / 4,320 | 0 / 0 | 4.000 | 152.799 ms | 261.388 ms | 1.013 FPS | 0 | throughput only |
| 17 + RT, run 1 | 60 s | 4,080 / 4,080 | 0 / 0 | 4.000 | 155.434 ms | 246.498 ms | 1.013 FPS | 0 | pass |
| 17 + RT, run 2 | 60 s | 4,080 / 4,080 | 0 / 0 | 4.000 | 155.709 ms | 255.867 ms | 1.013 FPS | 0 | reject tail |
| 16 + RT, run 1 | 60 s | 3,840 / 3,840 | 0 / 0 | 4.000 | 112.777 ms | 205.805 ms | 1.013 FPS | 0 | pass |
| 16 + RT, run 2 | 60 s | 3,840 / 3,840 | 0 / 0 | 4.000 | 112.151 ms | 285.131 ms | 1.013 FPS | 0 | reject tail |
| 15 + RT | 60 s | 3,600 / 3,600 | 0 / 0 | 4.000 | 119.561 ms | 416.198 ms | 1.013 FPS | 0 | reject tail |

Nineteen cameras fails on capacity while 18 cameras completes every scheduled
primary and PPE request, so 18 cameras plus one device-wide RT-DETR FPS is the
measured throughput boundary. It is not a supported production tier: reducing
the camera count did not monotonically remove rare primary latency spikes.
The Jetson remained in MAXN at approximately 67–69 degrees Celsius, with no
GPU, OOM, or throttling errors, so thermal pressure does not explain the tail.

The non-monotonic tail is consistent with unsynchronised work from the separate
TensorRT/CUDA context occasionally colliding with a primary batch. RT-DETRv4-S
must therefore enter the pipeline as a low-priority conditional phone-recall
specialist. However, the follow-up below proves that a simple exclusive lease
is not sufficient: low-priority work must never create a convoy of blocked
primary requests. Until a deadline-aware integration is implemented and
re-gated, the supported production capacity remains 21 cameras at 4 FPS
without RT-DETR.

## Exclusive low-priority admission rejection — 2026-07-13

A priority-aware reader/writer gate was implemented and tested as a Jetson
candidate. Existing primary and PPE requests remained concurrent readers. The
external RT-DETR context could acquire a crash-safe 125 ms writer lease only
while no primary request was active or waiting. A second version aligned lease
grants atomically to the instant the active primary set drained.

The local gate and remote-transport suite passed 81 tests. The candidate model
server loaded all three models, warmed both batch-4 TensorRT engines, and
completed a 21-camera no-RT-DETR control without drops. Mixed-load measurements
nevertheless rejected both lease policies:

| Policy | Cameras | Primary successes | Drops / failures | Minimum camera FPS | Primary p95 | Primary maximum | RT-DETR achieved | Skipped RT groups | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gate only, no RT control | 21 | 2,520 / 2,520 | 0 / 0 | 4.000 | 187.909 ms | 263.379 ms | — | — | throughput pass |
| Immediate idle lease | 18 | 4,320 / 4,320 | 0 / 0 | 4.000 | 175.981 ms | 387.407 ms | 0.933 FPS | 3 / 38 | reject |
| Primary-drain-aligned lease | 18 | 5,030 / 5,040 | 10 / 0 | 3.957 | 177.134 ms | 1,037.203 ms | 0.867 FPS | 4 / 30 | reject |

Mutual exclusion made each RT-DETR batch faster: its median dropped to about
31 ms because it no longer competed with primary CUDA work. It still made the
system worse. The immediate policy caused 26 primary waits averaging 31.1 ms,
with a 53.8 ms maximum. Drain alignment caused 26 waits averaging 34.4 ms.
When each lease released, the blocked primary requests woke together, creating
an inference burst; the aligned version eventually overloaded ten requests and
produced a one-second tail.

The candidate was rejected, its code was not shipped, and the proven batch-4
model server was restored. The next viable scheduler must use camera deadlines
or known phase gaps and must skip a conditional RT-DETR opportunity when the
remaining idle window is shorter than its measured runtime. It must not hold
or queue an already-admitted primary request behind specialist work.

## Phase-aware primary substitution — 2026-07-13

The next experiment replaced scheduled primary slots instead of adding an
independent specialist workload. Both benchmark processes used the same Jetson
monotonic timestamp. The edge harness omitted selected YOLO slots and counted
them as effective decisions only when the RT-DETR harness completed the exact
same number of frames. A verifier rejects count mismatches, overloads,
failures, stale RT groups, effective camera rates below 4 FPS, or primary
latency above 250 ms.

At 18 cameras, the grouped phase shape is `4+4+4+4+2`. Once every two seconds,
the final YOLO batch-2 slot was replaced by RT-DETRv4-S batch-2. At 20 cameras,
one four-camera phase was split into YOLO batch-2 plus RT-DETR batch-2. A PPE
pass that coincided with a substituted slot was deferred by one camera frame,
so total PPE coverage was preserved rather than dropped.

| Topology | Duration | YOLO + RT effective decisions | Drops / failures | Effective minimum camera FPS | PPE passes | Primary p95 | Primary maximum | RT-DETR achieved | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 18 cameras, remainder substitution, run 1 | 60 s | 4,260 + 60 = 4,320 | 0 / 0 | 4.000 | 468 | 133.987 ms | 209.138 ms | 1.000 FPS | pass |
| 18 cameras, remainder substitution, run 2 | 60 s | 4,260 + 60 = 4,320 | 0 / 0 | 4.000 | 468 | 128.284 ms | 223.588 ms | 1.000 FPS | pass |
| 20 cameras, split-phase substitution, run 1 | 60 s | 4,740 + 60 = 4,800 | 0 / 0 | 4.000 | 520 | 149.169 ms | 232.960 ms | 1.000 FPS | pass |
| 20 cameras, split-phase substitution, run 2 | 60 s | 4,740 + 60 = 4,800 | 0 / 0 | 4.000 | 520 | 153.003 ms | 234.691 ms | 1.000 FPS | pass |
| 21 cameras, batch-2 split probe | 30 s | 2,490 + 30 = 2,520 | 0 / 0 | 4.000 | 273 | 181.924 ms | 264.536 ms | 1.000 FPS | reject tail |
| 21 cameras, batch-1 singleton probe | 30 s | 2,490 + 30 = 2,520 | 0 / 0 | 4.000 | 273 | 175.492 ms | 265.010 ms | 1.000 FPS | reject tail |

All four passing 18- and 20-camera runs had zero stale RT-DETR groups. The
20-camera result is the current experimental conditional boundary. It is two
cameras better than additive coexistence and only one camera below the current
no-RT-DETR tier because the specialist consumes an existing detector slot
instead of creating new GPU demand.

This does not promote RT-DETR as the universal primary. Substitution should be
limited to phone-qualified person tracks and relevant camera profiles, with
tracking carrying context between frames. The labelled evidence establishes
better phone recall and person association, not complete vehicle and animal
parity. Production integration must route RT-DETR COCO records through the same
coordinate scaling, tracker, zone, rule, and alert contracts before this tier
can be deployed.

## Default-off model-server route — 2026-07-13

The RT-DETRv4-S batch-1 and batch-2 engines were then integrated behind
authenticated raw-frame model-server endpoints. The optional runtime is absent
unless both an engine path and its exact SHA-256 are configured. Startup warms
configured engines but a missing or invalid optional engine cannot take down
the existing YOLO model server. Runtime faults are fenced instead of reusing a
failed CUDA context. Only COCO person (`0`) and cell phone (`67`) records leave
the route; all other unvalidated classes are discarded before the normal record
contract.

The route remains default-off and is not called by the camera worker. That is
intentional: replacing a normal COCO frame with person/phone-only output could
otherwise falsely clear unrelated vehicle or animal rules. Worker activation
still requires rule-state carry-forward and phone-qualified track gating.

### Isolated route results

The same six positive and four negative labelled phone images were sent through
the actual FastAPI raw-frame endpoints, including request parsing, BGR-to-RGB
preprocessing, host-to-device copies, TensorRT, output copies, filtering, and
JSON serialization.

| Route | Actionable positives | Actionable negatives | Frame throughput | Median group latency | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RT-DETRv4-S batch-1 | 5 / 6 | 0 / 4 | 24.717 FPS | 38.086 ms | 44.119 ms | 44.526 ms |
| RT-DETRv4-S batch-2 | 5 / 6 | 0 / 4 | 28.170 FPS | 70.442 ms / 2 frames | 75.847 ms | 76.771 ms |

The HTTP/runtime wrapper is therefore a measured bottleneck: it reduces the
standalone batch-1 result from 38.424 to 24.717 FPS and batch-2 from 43.792 to
28.170 frame-FPS. A universal-primary deployment through this route would have
only a seven-camera arithmetic ceiling at 4 FPS before PPE duty. The earlier
direct-engine cadence test supported ten cameras and rejected eleven, but it
did not include the production transport overhead. Neither result supports
replacing the current YOLO primary.

### Production-shaped substitution through the route

The passing 20-camera split-phase topology was repeated for 60 seconds using
the actual model-server endpoint. Cameras 18 and 19 replaced one eighth of
their primary slots with RT-DETR batch-2, for exactly one aggregate RT-DETR
frame per second. The four-frame YOLO primary/PPE path and edge microbatch
transport were unchanged.

| Effective camera load | Effective decisions | PPE passes | Overloads / failures | Primary p95 | Primary maximum | RT-DETR achieved | RT stale groups | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 cameras at 4 FPS | 4,800 / 4,800 | 520 | 0 / 0 | 152.624 ms | 224.349 ms | 1.000 FPS | 0 | pass |

Under concurrent load the RT route itself measured 101.486 ms median, 167.641
ms p95, and 195.464 ms maximum per batch-2 group. It remained below the 250 ms
freshness budget and preserved the labelled 5/6 positive, 0/4 negative alert
outcome. The fail-closed substitution verifier passed with no errors.

This proves the server route and confirms the existing experimental boundary:
20 cameras at 4 effective decision FPS plus one device-wide RT-DETR FPS. It
does not change the supported live capacity, which remains 21 cameras at 4 FPS
without RT-DETR until camera-worker rule semantics and track gating are shipped
and re-gated.

## Production restoration

After the isolated builds and load tests:

- the model server returned healthy with three ready models and concurrent PPE
  batching enabled;
- the edge and model-server containers were restored;
- the one-minute model watchdog timer was restored and active;
- cam2 returned fresh on `gstreamer_nvdec` with zero inference overloads or
  failures;
- cam1 returned to its pre-existing source outage/reconnect state with zero
  inference overloads or failures;
- no RT-DETR engine or runtime configuration was promoted.

Raw build logs, engines, timing cache, and JSON benchmark evidence remain on
the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/rtdetrv4-s/`
