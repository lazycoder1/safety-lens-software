# Jetson 512px primary-router experiment — 2026-07-12

## Decision

Keep the YOLO26 Small primary at 640 pixels. Do not route full-size cameras
through the existing 512px Small TensorRT engine.

The 512px candidate loses person context that is required to enable the 960px
mobile-phone probe and was slower under the current concurrent production
runtime mix. It therefore fails both the accuracy and performance gates.

## Workload

Both production cameras remained live. Twenty images were replayed through the
real model-server endpoint at confidence 0.10, then evaluated at the production
0.30 confidence used for person, animal, and vehicle rules:

- thirteen existing phone/person positive and negative-control frames;
- four office NVR channel frames;
- three official COCO validation animal frames: two cat scenes and one dog
  scene.

Each resolution was run in both orders. The reported latency is the median of
the per-image three-repeat medians from the reverse-order run.

## Result

| Primary resolution | Median endpoint latency | Person-hit images | Animal-hit images | Vehicle-hit images | Phone-hit images |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 36.88 ms | 9 | 3 | 1 | 7 |
| 640 | 23.50 ms | 12 | 3 | 1 | 8 |

The animal gate retained both cat examples and the dog example at 512px, and
the motorcycle/vehicle control remained present. However, 512px lost the
production-threshold person in three images that 640px retained:

- `cam3-bench.jpg`;
- `neg-laptop.jpg`;
- `pos-dark.jpg`.

`pos-dark.jpg` is a phone-use positive. Its person confidence fell from 0.5649
at 640px to 0.1531 at 512px. Because phone-probe admission requires a primary
person, the 512px router would suppress the later 960px probe on that known
positive.

The 512px endpoint also ran 57% slower in this live mixed-runtime test. This is
the opposite of the isolated low-resolution-camera speedup and demonstrates
that an engine microbenchmark cannot be used as a full scheduling result when
the production low-resolution runtime is concurrently active.

## Guardrail

`scripts/benchmark_mobile_phone_jetson.py` now accepts `--imgsz` and reports
person, phone, animal, and vehicle confidence lists. Any future primary-router
resolution change must preserve production-threshold class presence, not only
raw model latency.

COCO validation sources:

- `http://images.cocodataset.org/val2017/000000039769.jpg`
- `http://images.cocodataset.org/val2017/000000255965.jpg`
- `http://images.cocodataset.org/val2017/000000482917.jpg`
