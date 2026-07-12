# Jetson YOLO26 Small 608-pixel primary experiment — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

Reject 608 pixels as the production YOLO26 Small primary resolution. The fixed
batch-4 INT8 engine improved isolated throughput by only 3.8%, from 105.113 to
109.063 FPS, and did not preserve the current primary detection contract across
the thirteen-frame office/phone corpus. No batch-2 engine was built and no
runtime or deployment setting was changed.

The production primary remains YOLO26 Small INT8 at 640 pixels. The supported
tiers remain 21 camera-equivalents at 4 FPS without RT-DETR, or 20 at 4
effective FPS plus one device-wide RT-DETRv4-S phone-recall FPS.

## Pre-screen

The PyTorch Small checkpoint was first evaluated at 640 and 608 pixels on six
labelled phone-use frames and four negative controls. Both resolutions produced
the same production phone-to-person outcomes:

- 4 / 6 actionable positive images;
- 0 / 4 actionable negative images.

That justified one batch-4 build, but phone parity alone was not treated as a
primary promotion gate.

## Candidate contract

- source SHA-256:
  `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`;
- input: fixed batch 4 at 608 by 608;
- precision: INT8;
- calibration YAML SHA-256:
  `2311ae61b75ec5029ff2dde49900baff0943163003825eb32d928a4557f860c7`;
- TensorRT: 8.5.2.2;
- engine SHA-256:
  `0ae412aec79cd9751ce11e31dd8ac6fa3b88cae28ba9e9ef2ac38847f899bb36`.

The build completed without OOM. TensorRT skipped two tactics that requested
1,109 MiB when 1,039 MiB of tactic workspace was available; the resulting
engine and manifest were otherwise valid.

## Isolated batch-4 result

Both engines used ten warmups and 200 timed groups over the same thirteen
office/phone frames at confidence 0.15.

| Engine | Throughput | Median / four frames | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Small 640 INT8 batch-4 | 105.113 FPS | 38.773 ms | 39.434 ms | 39.795 ms |
| Small 608 INT8 batch-4 | 109.063 FPS | 36.680 ms | 37.032 ms | 37.437 ms |

The candidate saved about 2.1 ms per four-frame group. This is useful but too
small to justify a new accuracy contract by itself.

## Primary-output rejection

The production primary owns person, vehicle, motorcycle, animal, phone, and
zone-routing context. At the live global confidence of 0.35, the 608 engine
changed retained class counts on eight of thirteen frames. Examples included:

- losing the person on the two copies of the dark office frame;
- losing the person on `neg-arm-table.jpg`;
- reducing retained motorcycles, cars, backpacks, and people on the worker
  frame while adding a handbag;
- changing retained phone and common-object counts on several phone controls.

The 608 engine preserved the labelled mobile-phone alert outcome, and its two
new cat-class proposals were below the live 0.35 threshold. It nevertheless
failed primary person/vehicle/object parity. The corpus is not fully labelled
for every COCO class, so there is no defensible basis for declaring the new
detections better than the current contract.

Because the isolated gain was only 3.8% and the primary output changed
materially, the experiment stopped before spending another offline build on a
batch-2 remainder engine or running a misleading camera-capacity gate.

## Restoration

The candidate engine was never configured in the model server. The original
model server and edge containers were restored with zero restart-count
increments, and the one-minute watchdog timer returned active.

Raw engine, manifest, build log, and benchmark evidence remain on the Jetson
under:

`/opt/rakshak-lens/model-server-models/coco_primary/experiments/int8-small/608/`
