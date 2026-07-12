# Jetson NVDEC output-drop experiment — 2026-07-12

## Decision

For the two verified 25 FPS office NVR feeds, configure Jetson
`nvv4l2decoder` to output every third decoded frame before VIC scaling and CPU
BGR conversion. The application still rate-limits delivery to 8 FPS.

The setting is opt-in and disabled by default. It must not be copied to a
lower-FPS source without measuring that source first; a 10 FPS camera with an
interval of three would be reduced below the application's 8 FPS target.

A follow-up capacity gate on 2026-07-13 sustained 20 repeated synthetic
960x540 H.264 streams plus the live cam2 worker for 60 seconds. All 20 test
pipelines stayed open and delivered at least 8.035 FPS. The current 20-camera
inference-compute tier is therefore below the demonstrated NVDEC ingest tier.

## Root cause

Both production streams advertise 25 FPS, while their configured processing
rate is 8 FPS. The previous GStreamer pipeline applied `videorate` after
`decodebin`, so NVDEC emitted 50 frames per second across the two cameras even
though only 16 frames per second were consumed. Scaling was already before CPU
BGR conversion, but all decoder outputs still reached the VIC path.

## A/B result

A duplicate live cam1 connection was sampled twice in each mode for 15 seconds
while the two production cameras remained active. Startup samples were removed
from the board-level averages.

| Mode | Delivered FPS | Capture-process CPU | Board CPU | VIC | Input power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decoder output drop disabled | 8.18 / 8.17 | 5.62–5.67% | 5.09% | 24.26% | 8.06 W |
| Every third decoder output | 8.10 / 8.15 | 4.69–4.81% | 4.93% | 17.05% | 8.03 W |

The change reduced capture-process CPU by about 16% and VIC utilization by
about 30%, with no material delivered-FPS or power change. Cam2 also delivered
8.14 FPS through the optimized path at its native 352x288 resolution.

## Concurrent ingest validation

The load test retained both production workers and opened additional live RTSP
connections to the two real NVR feeds. These are real decoder/network
pipelines, but repeated sources rather than twelve distinct camera scenes.

| Simultaneous RTSP/NVDEC pipelines | Test-stream delivered FPS | Board CPU | VIC | Input power | Production inference drops / failures |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 8.09–8.17 | 7.19% | 12.93% | 8.37 W | 0 / 0 |
| 10 | 8.06–8.17 | 25.32% | 20.40% | 8.96 W | 0 / 0 |
| 12 | 8.04–8.17 | 31.93% | 25.37% | 9.18 W | 0 / 0 |

All twelve pipelines remained open and met the 8 FPS ingest target. Both
production cameras remained frame-fresh on `gstreamer_nvdec`, with zero capture
failures. Alert persistence, callback, delivery, and backpressure failures also
remained zero.

## 2026-07-13 capacity follow-up

The reusable benchmark was extended to open and read up to 32 capture copies
concurrently. The first sweep repeated the currently healthy 352x288 cam2 NVR
feed while its production worker remained active.

| Added test streams | Total NVDEC pipelines | Minimum / median test FPS | Aggregate test FPS | Capture-process CPU | Board CPU | VIC | Input power |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 13 | 7.913 / 7.977 | 95.923 | 28.38% | 6.18% | 6.00% | 7.66 W |
| 16 | 17 | 7.962 / 8.027 | 128.304 | 38.02% | 9.95% | 9.14% | 7.76 W |
| 20 | 21 | 7.925 / 7.990 | 159.729 | 47.08% | 9.15% | 10.43% | 7.92 W |

To avoid treating a low-resolution feed as evidence for the requested
approximately 900-pixel operating point, a local credential-free RTSP source
was generated at 960x540, 25 FPS, H.264. MediaMTX fanned that single source out
to distinct production `GStreamerCapture` and `nvv4l2decoder` pipelines. Each
pipeline applied decoder interval three and the normal 8 FPS output policy.

| Added 960x540 streams | Total NVDEC pipelines | Duration | Minimum / median test FPS | Aggregate test FPS | Capture-process CPU | Board CPU | VIC | Input power | Maximum RAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 13 | 15 s | 8.079 / 8.079 | 97.213 | 47.87% | 8.17% | 17.75% | 8.27 W | 5,381 MB |
| 16 | 17 | 15 s | 8.074 / 8.074 | 129.249 | 63.40% | 11.77% | 19.44% | 8.60 W | 5,487 MB |
| 20 | 21 | 60 s | 8.035 / 8.040 | 160.806 | 78.50% | 18.23% | 27.98% | 8.96 W | 5,615 MB |

The 60-second confirmation delivered 483 frames on every synthetic stream.
RAM remained between 5,562 and 5,615 MB. Cam2 stayed frame-fresh on
`gstreamer_nvdec` throughout, with zero inference overloads or failures. Cam1
was already in a source-side outage and was not counted as an active pipeline.

The synthetic clients can join between H.264 keyframes, which caused transient
decoder reference/surface warnings during startup. Every client recovered,
opened successfully, and sustained the target cadence; no warning persisted as
a read failure.

## Capacity interpretation

Ingest is not the current camera-count limiter for the measured workload. The
21-pipeline 960x540 NVDEC result clears the current conditional inference tier
of 20 camera-equivalents at 4 detection FPS with YOLO26 Small primary inference
and 11.1% PPE specialist duty.

The supported planning numbers remain:

- 20 camera-equivalents at 4 primary detection FPS for the measured conditional
  Small-model workload;
- 21 concurrent 960x540 RTSP/NVDEC pipelines demonstrated at 8 FPS, consisting
  of 20 repeated synthetic streams plus one live NVR stream;
- 18 camera-equivalents at 4 FPS when one aggregate RT-DETRv4-S phone-recall
  specialist FPS is enabled.

This is an engineering capacity result, not an accuracy SLA or proof of 21
unique production cameras. Repeating one local RTSP source proves concurrent
decoder, scaler, BGR-conversion, and capture-thread capacity without proving
independent network paths, NVR session limits, source jitter, or long-duration
reconnect behavior. The system planning cap therefore remains the lower
20-camera inference tier, and a deployment should still soak its actual camera
mix before licensing that count.

## Implementation

`SAFETYLENS_NVDEC_DROP_FRAME_INTERVAL` is bounded to the hardware-supported
range of 0–30. The callback only writes `drop-frame-interval` on an actual
`nvv4l2decoder` while the element is still in NULL/READY state. The reusable
benchmark prints camera IDs and metrics but never prints RTSP credentials. It
now supports a credential-safe environment URL override, concurrent copy
counts, aggregate and per-stream delivered rates, frame dimensions, and
process CPU. Each capture is timed independently so a slow final read or
release on one worker cannot under-report every stream's FPS.

Raw follow-up evidence is retained on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/nvdec-960x540-20260713/`
