# Jetson primary batch-8 experiment — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

Reject batch-8 as a production primary profile and keep the proven batch-4
YOLO26 Small INT8 pipeline. Batch-8 improved isolated equal-work throughput by
10.0%, but it did not increase the supported camera tier. It raised end-to-end
tail latency, introduced remainder timeouts, and made the existing 20-camera
gate less reliable.

No batch-8 runtime, route, environment variable, or deployment profile was
promoted. The supported conditional-inference tier remains 20 camera
equivalents at 4 AI FPS per camera with 11.1% PPE specialist duty.

## Question

Test whether widening the primary TensorRT microbatch from four to eight frames
can increase camera capacity without changing the Small-model accuracy
contract. The experiment intentionally uses YOLO26 Small, not Nano.

## Engine contract

- source model SHA-256:
  `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`;
- calibration corpus/config digest:
  `2311ae61b75ec5029ff2dde49900baff0943163003825eb32d928a4557f860c7`;
- engine: fixed batch 8, 640-pixel input, INT8;
- engine SHA-256:
  `ce1caa74891828e374c7bf58b763817e7fe93133f1a1135e74058de17130cc4f`;
- engine size: 11,303,519 bytes;
- TensorRT context allocation: approximately 111 MiB;
- build time: 557 seconds.

The engine reused the same Small checkpoint and calibration contract as the
promoted batch-4 primary. This isolates the scheduling/batch-size change from
model and calibration changes.

## Isolated equal-work result

Both engines processed groups of eight frames from the same 13-frame Jetson
office/phone corpus. Batch-4 used two engine calls per group; batch-8 used one.
The timed workload contained 1,600 frames.

| Engine | Throughput | Median per 8 frames | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Batch-4 INT8 | 104.769 FPS | 75.473 ms | 78.813 ms | 79.136 ms |
| Batch-8 INT8 | 115.264 FPS | 68.311 ms | 71.626 ms | 72.202 ms |

All 77 detections were bit-for-bit identical: same classes, boxes, and
confidences, minimum IoU 1.0, and maximum confidence drift 0. The candidate
therefore passed the isolated speed and output-parity checks.

## System gate

The exact edge load exercised 640-pixel YOLO26 Small INT8 primary inference at
4 FPS per camera, 11.1% YOLOE-26S PPE duty, grouped camera phases, bounded
remote admission, and the batch-8 primary route. The freshness requirement is
that maximum latency stay below the next 250 ms camera inference period, with
zero drops, failures, and overloads.

| Cameras | Duration | Completed | Drops / overloads | p95 | Maximum | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 30 s | 2,880 / 2,880 | 0 / 0 | 248.812 ms | 364.788 ms | reject |
| 22 | 30 s | 2,640 / 2,640 | 0 / 0 | 243.215 ms | 352.696 ms | reject |
| 21 | 30 s | 2,520 / 2,520 | 0 / 0 | 222.745 ms | 339.092 ms | reject |
| 20 | 30 s | 2,399 / 2,400 | 1 / 1 | 212.666 ms | 330.219 ms | reject |
| 20, two admission slots | 10 s | 800 / 800 | 0 / 0 | 173.535 ms | 249.691 ms | no capacity gain |
| 20, three admission slots | 10 s | 800 / 800 | 0 / 0 | 187.536 ms | 287.798 ms | reject |

At 24 cameras, increasing the remainder-flush delay from 12 ms to 16 ms and
20 ms made the tail worse: p95 rose from 248.812 ms to 259.733 ms and
268.765 ms, while maximum latency stayed above 350 ms. A separate eight-camera
route smoke did prove that full groups use batch-8 correctly, completing
320/320 requests at 4 FPS with zero fallbacks, overloads, or failures.

The pre-candidate batch-4 control at 24 cameras completed 2,877/2,880 requests,
with three overloads, 215.377 ms p95, and 293.575 ms maximum. It confirms that
24 cameras is beyond the current tier and that aligning larger groups does not
create hidden capacity.

## Why isolated FPS did not become camera capacity

Batch-8 makes each primary engine call faster per frame, but it changes the
arrival shape from smaller continuous work into larger bursts. At the global
camera phase boundary, primary batches and PPE work contend at the same time.
Remainders also wait or split into batch-4/batch-2 calls. That contention and
queueing dominate the roughly 7 ms saved per eight primary frames.

This result is an important scheduling constraint: model-only FPS is not a
sufficient promotion metric. Camera capacity must be decided by the full
primary, specialist, transport, admission, and freshness gate.

## Validation and restoration

- 115 focused runtime, transport, scheduler, and compose tests passed in a
  disposable Jetson test environment;
- the candidate containers were removed from the live names after the failed
  system gate;
- the original batch-4 edge and model-server images were restored with NVIDIA
  runtime;
- the edge was restarted after the model server so transient failures from the
  first restore order were cleared;
- cam2 returned fresh on `gstreamer_nvdec` with zero overloads and failures;
- cam1 remained subject to its pre-existing source availability state.

Raw engine, parity, and load-test evidence is stored on the Jetson under:

`/opt/rakshak-lens/model-server-models/coco_primary/experiments/int8-small/batch8/`
