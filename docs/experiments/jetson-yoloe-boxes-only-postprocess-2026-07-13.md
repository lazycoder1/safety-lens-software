# Jetson YOLOE boxes-only postprocessing experiment — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

Reject custom boxes-only postprocessing for the YOLOE-26S PPE engines. The
alert pipeline consumes boxes, classes, and confidence but not instance masks,
so avoiding mask construction is locally faster on PPE-positive frames.
However, the exact mixed-camera workload gained no camera tier and produced a
worse 20-camera tail than the proven runtime. No custom predictor or runtime
flag was promoted.

The supported tier remains 20 camera equivalents at 4 AI FPS per camera with
11.1% PPE specialist duty.

## Candidate

Ultralytics' segmentation predictor runs NMS, constructs an instance mask for
each retained YOLOE proposal, filters proposals whose masks are empty, and
then returns boxes plus masks. SafetyLens immediately converts those results
to box records and discards the masks.

The candidate retained the same TensorRT engine and preprocessing/NMS path but
overrode only result construction:

1. scale the already-retained boxes to source coordinates;
2. return the first six box columns;
3. do not materialize masks.

This was intentionally an opt-in candidate because omitting empty-mask
filtering could theoretically retain a box the default segmentation path
would discard.

## Curated PPE result

The fixed batch-4 FP16 engine processed the 18-image PPE validation corpus at
the production 0.20 confidence. Both execution orders used ten warmups and 200
timed groups, or 800 timed frames per mode.

| Order | Default throughput | Boxes-only throughput | Gain | Default median | Boxes-only median | Detection parity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default first | 70.591 FPS | 73.806 FPS | 4.55% | 56.614 ms | 54.808 ms | exact, 14/14 |
| Boxes first | 70.680 FPS | 73.963 FPS | 4.65% | 56.614 ms | 54.717 ms | exact, 14/14 |

Reported postprocessing fell from approximately 1.25 to 0.63 ms per frame.
Inference and preprocessing were unchanged. The default path constructed masks
for 267 of 800 timed results; the candidate constructed none.

## Expanded parity

The comparison was repeated across 274 images from the office/PPE calibration
corpus. At the production threshold, both paths returned exactly the same 31
detections across every unique input frame: identical class IDs, confidences,
and boxes. No empty-mask-only box appeared.

Most expanded-corpus frames contained no PPE detection. In that sparse mix,
throughput was effectively unchanged: 74.310 FPS default versus 74.465 FPS
boxes-only. This demonstrates that the saving scales with retained-mask count,
not total camera throughput.

## Exact edge-load gate

The candidate model server ran the real edge transport with 640-pixel YOLO26
Small INT8 primary inference, batch-4 phases, 4 FPS per camera, 11.1% YOLOE
PPE duty, deferred phone/PPE overlap, four bounded admission slots, and a
125 ms admission wait.

| Cameras | Duration | Completed | Drops / failures | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 21 | 30 s | 2,520 / 2,520 | 0 / 0 | 178.051 ms | 254.306 ms | reject: stale tail |
| 20 | 60 s | 4,800 / 4,800 | 0 / 0 | 144.745 ms | 208.165 ms | pass tier, no improvement |

The proven default 20-camera cold gate recorded 141.233 ms p95 and 171.341 ms
maximum. The boxes-only candidate therefore did not improve the authoritative
mixed workload despite its repeatable PPE-positive microbenchmark gain.
Twenty-one also crossed the 250 ms camera inference period and cannot be
promoted.

## Interpretation

PPE postprocessing is not the current capacity bottleneck. Primary inference,
preprocessing, cross-camera remainder handling, and queue shape dominate the
mixed workload. A native detection-format PPE model could still be useful if
it is accuracy-gated, but a custom segmentation predictor adds a maintenance
branch without measurable system benefit.

The original batch-4 model server was restored and reported both primary and
PPE batch-2/batch-4 TensorRT engines warmed. Cam2 returned fresh on
`gstreamer_nvdec` with zero overloads or failures; cam1 remained in its
pre-existing source outage.

Raw benchmark and load evidence is stored on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/yoloe-boxes-only/`
