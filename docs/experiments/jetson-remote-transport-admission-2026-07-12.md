# Jetson remote transport and admission experiment — 2026-07-12

## Decision

Keep raw BGR transport enabled for the local edge-to-model-server hop and raise
the bounded remote admission wait from 50 ms to 65 ms.

The production inference scheduler deterministically phases enabled cameras
across their inference interval. Under that schedule, 65 ms removed the
remaining ten-camera admission drops while staying below half of the 125 ms
stale-slot threshold at 4 FPS. Eleven cameras still shed overload, so the wait
does not turn sustained excess demand into an unbounded queue.

## Load shape

The two production RTSP cameras remained live. Virtual cameras replayed four
representative office frames through YOLO26 Small at 4 FPS, requested a 960px
phone probe once per second, and invoked the Small PPE specialist at 11.1% duty.
The client and production edge both used a two-request admission bound.

## Raw versus JPEG

At seven virtual plus two live cameras, both transports completed all 840
scheduled virtual requests:

| Transport | Drops | Median | p95 |
| --- | ---: | ---: | ---: |
| Raw BGR | 0 | 22.323 ms | 52.019 ms |
| JPEG quality 85 | 0 | 24.156 ms | 52.841 ms |

At eight virtual plus two live cameras with the original 50 ms admission wait:

| Transport | Drops | Median | p95 |
| --- | ---: | ---: | ---: |
| Raw BGR | 3 / 960 | 23.127 ms | 52.256 ms |
| JPEG quality 85 | 4 / 960 | 24.393 ms | 56.531 ms |

Raw transport is a modest but consistent win on this same-host Docker topology.
JPEG remains the rolling-upgrade fallback.

## Admission boundary

With raw transport and the production staggered phase model:

| Total camera-equivalents | Admission wait | Completed | Drops | Minimum FPS | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 50 ms | 957 / 960 | 3 | 3.933 | 52.256 ms |
| 10 | 65 ms | 960 / 960 | 0 | 4.000 | 50.553 ms |
| 11 | 65 ms | 1074 / 1080 | 6 | 3.933 | 64.998 ms |

The zero-drop Small-model compute boundary therefore moves from nine to ten
camera-equivalents at 4 FPS for the tested workload. This is not a claim that
ten simultaneous RTSP decoders have been validated; four live RTSP/NVDEC
pipelines remain the separately measured ingest result.

## Burst limit

Paired synthetic arrivals caused five drops at 65 ms and two at 80 ms. The
runtime does not use that arrival policy: `inference_scheduler` assigns each
enabled camera a stable phase within the interval. The paired result is retained
as evidence that the phase scheduler is part of the capacity contract and that
simply increasing the wait further would be the wrong fix.
