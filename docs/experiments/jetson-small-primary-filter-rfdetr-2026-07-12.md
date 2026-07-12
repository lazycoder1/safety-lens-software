# Jetson Small primary-filter and RF-DETR experiment — 2026-07-12

## Decision

Promote the calibrated YOLO26 Small INT8 640px engine as the only full-frame
COCO path. Retire the periodic 960px FP16 mobile-phone probe unless future
site-labelled footage proves that it recovers actionable cases.

Keep the existing rule-compiled cascade:

1. decode each camera with NVDEC;
2. run the 640px Small COCO primary at the configured inference cadence;
3. apply person geometry and configured PPE zones to the primary result;
4. run PPE or other specialists only when their rule, schedule, and primary
   context make an actionable result possible;
5. evaluate phone detections only when they associate with a person;
6. defer a PPE pass by one inference frame when it would collide with another
   scheduled specialist burst.

This is a Small-only result. Nano was not used.

## Labelled mobile-phone accuracy

Six phone-use positives and four negative controls were evaluated at confidence
0.10, then judged with the production 0.15 phone threshold, 0.30 person
threshold, and phone-to-person association geometry.

| Candidate | Positive cases | Actionable negatives | Median inference |
| --- | ---: | ---: | ---: |
| YOLO26 Small INT8, full frame 640 | 4 / 6 | 0 / 4 | 13.85 ms |
| YOLO26 Small FP16, full frame 960 | 4 / 6 | 0 / 4 | 30.13 ms |
| YOLO26 Small INT8, 640 full frame plus person crop | 4 / 6 | 1 / 4 | about 27.3 ms total |
| RF-DETR Small FP16, full frame 512 | 5 / 6 | 1 / 4 | 32.20 ms GPU |

The larger YOLO input did not recover a labelled case. The person crop doubled
work without improving recall and strengthened a negative phone score. RF-DETR
Small recovered one extra positive but also introduced a strong actionable
negative and used about 2.3 times the deployed INT8 engine's GPU time.

## RF-DETR deployment result

RF-DETR Small 1.8.3 was exported from the official COCO checkpoint. Its default
opset-17 ONNX could not parse on the Jetson's TensorRT 8.5 because that runtime
lacks the emitted `LayerNormalization` operator. Re-exporting with opset 16
decomposed that operation and produced a working FP16 engine.

- export resolution: 512x512;
- TensorRT build time: 345.571 seconds;
- engine size: 56 MiB;
- build-time activation peak reported by TensorRT: about 2.18 GiB;
- isolated engine throughput: 31.05 FPS;
- isolated median GPU latency: 32.20 ms.

The build skipped tactics requesting more device memory than was available.
RF-DETR therefore fails both the latency and operational-complexity gates for
this 8 GB Orin, even before integrating its different preprocessing,
postprocessing, and class-ID contract into the model server.

## Tracker result

ByteTrack was tested as a filter between the primary detector and specialists.
It added about 0.81 ms median processing, worsened the tail, and produced sparse
track IDs on the low-FPS office samples. It did not create a safe opportunity
to skip additional primary frames or specialist evaluations. It is rejected as
a capacity optimization. Tracking remains appropriate for dwell, queue, and
identity-stability features when those features themselves require it.

The production filter instead uses the previous fresh COCO result plus the same
person-size, confidence, vehicle-association, and zone geometry enforced by the
alert policy. This avoids running a tracker solely to answer a boolean
"can this specialist produce an actionable result?" question.

## Exact mixed-load capacity

The capacity workload used staggered cameras at 4 FPS, calibrated INT8 Small
640px primary inference, 11.1% PPE specialist duty, one-second phone scheduling,
three admission slots and deferred phone/PPE overlap. The initial 65 ms result
was followed by a bounded admission sweep after the 640-only workload removed
the high-resolution phone burst.
All specialist evaluations were preserved.

| Cameras | Run | Completed | Drops / failures | Minimum FPS | Specialist calls | Median | p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | A | 1560 / 1560 | 0 / 0 | 4.000 | 169 | 17.506 ms | 69.537 ms |
| 13 | B | 1560 / 1560 | 0 / 0 | 4.000 | 169 | 17.505 ms | 68.441 ms |
| 14 | 65 ms boundary | 1670 / 1680 | 10 / 0 | 3.767 | 182 | 17.571 ms | 71.520 ms |
| 14 | 75 ms A | 1680 / 1680 | 0 / 0 | 4.000 | 182 | 17.529 ms | 71.424 ms |
| 14 | 75 ms B | 1680 / 1680 | 0 / 0 | 4.000 | 182 | 17.484 ms | 70.876 ms |
| 15 | 75 ms boundary | 1775 / 1800 | 25 / 0 | 3.733 | 190 | 17.953 ms | 72.247 ms |

Fourteen conditional cameras at 4 inference FPS is the supported tier for this
workload. Fifteen is the measured failure boundary. The 75 ms wait remains well
below the 250 ms frame period and is bounded to prevent sustained excess demand
from becoming an unbounded queue. This replaces the prior eleven-camera tier
that retained a 960px FP16 phone burst.

This is not a claim that every mix of fourteen cameras is safe. A workload that
runs PPE on every person frame, enables pose or other specialists everywhere,
or uses denser phone rules must be benchmarked separately. The conservative
all-person planning tier remains lower than the conditional tier.

## Fifteen-camera concurrency follow-up

Increasing the bounded work queue does not prove a fifteenth camera. The direct
HTTP harness initially made four slots and a 100 ms wait look clean, but the
exact edge transport exposed the extra session, admission, and transport tail.

| Shape | Completed | Drops | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 slots, 75 ms, direct | 1775 / 1800 | 25 | 17.953 ms | 72.247 ms | 92.790 ms |
| 4 slots, 75 ms, direct | 1788 / 1800 | 12 | 18.109 ms | 92.157 ms | 111.209 ms |
| 4 slots, 100 ms, direct | 1800 / 1800 | 0 | 18.030 ms | 94.674 ms | 112.322 ms |
| 4 slots, 100 ms, exact edge | 1799 / 1800 | 1 | 18.398 ms | 127.814 ms | 181.381 ms |

The exact path failed the zero-drop gate and crossed half of the 250 ms frame
period at p95. Production therefore remains at three slots and 75 ms. The
benchmark gained `--edge-url-override` so an isolated container can exercise
`model_manager.predict_record_batches()` without mounting production camera
configuration or copying its secrets.
