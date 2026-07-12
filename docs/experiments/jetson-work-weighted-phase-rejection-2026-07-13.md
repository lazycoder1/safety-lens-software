# Jetson work-weighted camera phase experiment — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

Reject work-weighted phase spacing as a production capacity change. Reserving
less of each 250 ms inference period for the cheaper final batch-2 remainder
removed overload drops in the 22-camera probes and reduced the worst cold tail,
but neither tested weight kept maximum primary latency below the 250 ms
freshness deadline.

The production scheduler remains evenly spaced. The supported tiers remain 21
camera-equivalents at 4 FPS without RT-DETR, or 20 at 4 effective FPS plus one
device-wide RT-DETRv4-S phone-recall FPS.

## Service-cost measurement

The current edge transport was measured through the live model-server routes
before changing the benchmark schedule. Each scenario used five warmups and 40
timed groups.

| Work group | Median | p95 | Maximum | Saturation throughput |
| --- | ---: | ---: | ---: | ---: |
| Primary singleton | 16.180 ms | 17.260 ms | 25.580 ms | 60.485 FPS |
| Primary batch-2 | 30.570 ms | 31.689 ms | 44.184 ms | 64.209 frame-FPS |
| Primary batch-4 | 37.900 ms | 40.191 ms | 49.858 ms | 104.068 frame-FPS |
| Primary + PPE batch-2 | 50.445 ms | 54.301 ms | 66.096 ms | 38.972 frame-FPS |

The 22-camera topology is five batch-4 groups plus one batch-2 group. Uniform
spacing gives every group 41.667 ms. A remainder weight below one reallocates
part of the batch-2 interval to the five full groups without changing model
work, FPS, confidence, resolution, or PPE duty.

## Exact edge-load sweep

The production edge was stopped during the isolated sweep so its own camera
workers could not bypass the benchmark admission semaphore. The workload used
YOLO26 Small INT8 at 640 pixels, 22 cameras at 4 FPS, 11.1% YOLOE-26S PPE duty,
batch-4 phases, four admission slots, a 125 ms admission bound, and the current
14 ms/6 ms adaptive microbatch timings.

| Final-group weight | Completed | Drops / failures | Minimum FPS | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 uniform cold control | 2,636 / 2,640 | 4 / 0 | 3.967 | 188.352 ms | 379.670 ms | reject |
| 0.8 | 2,640 / 2,640 | 0 / 0 | 4.000 | 186.426 ms | 278.252 ms | reject tail |
| 0.7 | 2,640 / 2,640 | 0 / 0 | 4.000 | 187.849 ms | 271.165 ms | reject tail |

Both weighted runs converted overload into completed work and improved the
maximum relative to the cold control, but the best maximum was still 21.165 ms
late. A deeper queue or a more aggressive weight would only move lateness to
the next cycle; the measured batch-2 p95 already consumes about 31.7 ms.
Increasing the supported tier requires less inference work or faster engines,
not another phase-spacing heuristic.

## Reproducibility and restoration

`scripts/benchmark_conditional_model_server_load.py` now exposes
`--phase-remainder-weight` for future scheduler experiments. The default is
1.0, so existing commands and production-shaped controls remain unchanged.

The disposable benchmark containers were removed. The live edge container and
one-minute watchdog timer were restored; the edge and model server both
reported running with zero container restarts.

Raw evidence is stored on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/weighted-phase/`
