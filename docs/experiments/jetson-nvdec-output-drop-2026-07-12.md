# Jetson NVDEC output-drop experiment — 2026-07-12

## Decision

For the two verified 25 FPS office NVR feeds, configure Jetson
`nvv4l2decoder` to output every third decoded frame before VIC scaling and CPU
BGR conversion. The application still rate-limits delivery to 8 FPS.

The setting is opt-in and disabled by default. It must not be copied to a
lower-FPS source without measuring that source first; a 10 FPS camera with an
interval of three would be reduced below the application's 8 FPS target.

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

## Capacity interpretation

Ingest is no longer the current camera-count limiter: twelve concurrent live
decoder pipelines clear the separately measured conditional inference boundary
of eleven camera-equivalents at 4 detection FPS.

The supported planning numbers remain:

- 11 cameras at 4 detection FPS for the measured workload where person context
  enables the high-resolution phone probe in 25% of slots;
- 10 cameras at 4 detection FPS for the conservative all-person context case;
- 12 concurrent RTSP/NVDEC ingest pipelines demonstrated at 8 FPS, using two
  unique NVR streams repeated under load.

This is an engineering capacity result, not an accuracy SLA. YOLO26 Small
remains the deployed primary model. RT-DETR-L was not promoted because its
640-pixel median latency was 4.7 times slower and its TensorRT exports exceeded
the device's available unified memory.

## Implementation

`SAFETYLENS_NVDEC_DROP_FRAME_INTERVAL` is bounded to the hardware-supported
range of 0–30. The callback only writes `drop-frame-interval` on an actual
`nvv4l2decoder` while the element is still in NULL/READY state. The reusable
benchmark prints camera IDs and metrics but never prints RTSP credentials.

Deployed edge revision: `c05a474`.
