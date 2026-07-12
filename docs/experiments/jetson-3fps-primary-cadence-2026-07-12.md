# Jetson 3 FPS primary-cadence experiment — 2026-07-12

## Decision

Reject lowering the primary detector cadence from 4 FPS to 3 FPS as a capacity
optimization. It did not create a new zero-drop camera tier, so the slower
detection and alert-confirmation cadence has no compensating throughput gain.

Retain the demonstrated eleven-camera conditional limit at 4 detection FPS
with YOLO26 Small.

## Method

The production edge process was stopped for each 30-second run and restored by
a detached watchdog. A standalone edge process exercised the exact
`model_manager.predict_record_batches()` path with:

- YOLO26 Small primary inference at 3 FPS;
- 640px normal primary frames;
- one 960px phone probe per camera per second;
- the Small PPE specialist at 11.1% deterministic duty;
- the production staggered phase, three inference slots, 65 ms bounded
  admission, raw BGR transport, and prewarmed thread-local sessions.

Four representative office frames were replayed. The model server was not
changed. Both live cameras returned to fresh GStreamer/NVDEC operation with
zero capture or inference failures after the guarded runs.

## Results

| Cameras | Run | Completed | Drops / failures | Minimum FPS | Median | p95 | Maximum |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | A | 1078 / 1080 | 2 / 0 | 2.933 | 20.968 ms | 107.566 ms | 152.737 ms |
| 12 | B | 1076 / 1080 | 4 / 0 | 2.933 | 21.075 ms | 111.632 ms | 178.188 ms |
| 13 | A | 1153 / 1170 | 17 / 0 | 2.733 | 21.255 ms | 123.003 ms | 187.231 ms |
| 14 | A | 1237 / 1260 | 23 / 0 | 2.733 | 21.150 ms | 117.799 ms | 174.225 ms |

Twelve failed both repeats, so 3 FPS does not improve the clean supported
count over the existing eleven-camera 4 FPS result.

## Bottleneck interpretation

Reducing nominal primary cadence does not reduce every expensive operation:
the high-resolution phone recovery probe remains fixed at 1 Hz per contextual
camera. At higher camera counts, the 960px probe and service-time tail consume
the saved 640px primary budget and cause bounded admission loss. The rising
p95 and maximum latency confirm contention rather than request failures.

The camera planner also emits `tracking_enabled`, but the live video path has
no object tracker. The only tracker object in that loop monitors RTSP
connection outages. Zone evaluation consumes completed detector results after
inference. The deployed shape is therefore:

`primary detector -> previous-detection context gate -> specialists -> zones`

It is not yet the intended:

`primary detector -> object tracker -> zones -> conditional specialists`

A lower detector cadence should not be reconsidered until a real object
tracker is implemented and validated for temporal recall, zone crossings, and
alert timing. The follow-up phone-probe experiment found that a true 832px
engine and 640px person crops preserved the small phone corpus, but neither
produced a safe new camera tier in the current full-frame, tracker-free
architecture.
