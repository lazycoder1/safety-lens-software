# Jetson partial-phase weighting for PPE substitution — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

> **Physical-capacity note:** the 27-camera result below is an inference-only
> camera-equivalent tier. Combined RTSP/NVDEC and inference testing later
> established 25 as the repeatably validated physical tier. See
> `jetson-physical-camera-capacity-2026-07-13.md`.

## Decision

Use a 0.70 final-phase weight for the Jetson batch-4 scheduler. A partial last
phase costs less than a full four-camera phase, so uniform spacing leaves too
little room between the preceding full GPU batches. The weighted schedule
reserves 0.70 of a normal phase interval after the remainder and redistributes
the saved time across the preceding full phases.

This raises the validated inference tier from 24 to 27 camera-equivalents at
four effective decisions per second. The strict gate also ran PPE at one FPS
on every camera simultaneously and retained one device-wide RT-DETRv4-S FPS.
Two 60-second runs completed every decision with zero overloads, failures,
route fallbacks, or stale detector groups.

This is an inference-capacity result, not evidence that 27 physical RTSP
decoders, network streams, and alert workloads have completed a simultaneous
soak. The production camera count must remain bounded separately by capture,
memory, and source reliability.

## Bottleneck

The earlier 25-camera sweep had one 282.405 ms PPE tail above the 250 ms camera
period. Twenty-five cameras form six full batch-4 phases plus one singleton.
Twenty-six adds a batch-2 remainder, and 27 adds a three-camera remainder. The
remainder work is cheaper than a full phase, but equal phase spacing causes it
to collide with the full PPE/primary work ahead of it.

The 28-camera topology is seven complete batch-4 phases, so remainder weighting
cannot change it. Its normal 0.5 PPE-FPS gate already reached 293.937 ms and is
therefore the current hard rejected inference boundary.

## Weight sweep

Every 30-second run completed all expected decisions at exactly four effective
FPS per camera with zero overloads or failures.

| Cameras | Final-phase weight | Effective decisions | Primary maximum | PPE maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 25 | 1.00 | 3,000 / 3,000 | 111.409 ms | 208.196 ms | control pass |
| 25 | 0.85 | 3,000 / 3,000 | 103.299 ms | 197.144 ms | pass |
| 25 | 0.70 | 3,000 / 3,000 | 99.973 ms | 189.254 ms | pass |
| 25 | 0.55 | 3,000 / 3,000 | 100.023 ms | 188.873 ms | pass |
| 26 | 0.85 | 3,120 / 3,120 | 104.727 ms | 201.232 ms | pass |
| 26 | 0.70 | 3,120 / 3,120 | 107.005 ms | 197.289 ms | pass |
| 26 | 0.55 | 3,120 / 3,120 | 111.572 ms | 193.393 ms | pass |
| 27 | 0.85 | 3,240 / 3,240 | 157.333 ms | 210.803 ms | pass |
| 27 | 0.70 | 3,240 / 3,240 | 162.887 ms | 206.606 ms | select |
| 27 | 0.55 | 3,240 / 3,240 | 160.209 ms | 206.720 ms | pass |

The selected 0.70 value is not the lowest tail in every smaller topology, but
it was the best 27-camera result and retained ample margin at 25 and 26. Full
groups ignore the setting, and malformed, non-finite, or out-of-range values
fall back or clamp to the bounded 0.10–1.00 range.

## Combined RT-DETR and confirmation gates

The RT pair occupied one full camera phase. It substituted one aggregate
primary frame per second, while PPE work was deferred rather than discarded
on collisions.

| Cameras | PPE target | Duration | Effective decisions | PPE frames | RT frames | Primary maximum | PPE maximum | RT maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.5 FPS | 30 s | 3,000 / 3,000 | 375 | 30 | 104.285 ms | 188.310 ms | 74.682 ms |
| 26 | 0.5 FPS | 30 s | 3,120 / 3,120 | 390 | 30 | 130.520 ms | 217.993 ms | 100.186 ms |
| 27 | 0.5 FPS | 30 s | 3,240 / 3,240 | 405 | 30 | 181.471 ms | 214.897 ms | 79.136 ms |
| 27 | 1.0 FPS | 30 s | 3,240 / 3,240 | 810 | 30 | 164.015 ms | 208.567 ms | 67.750 ms |
| 27, run 1 | 1.0 FPS | 60 s | 6,480 / 6,480 | 1,620 | 60 | 197.877 ms | 226.325 ms | 91.132 ms |
| 27, run 2 | 1.0 FPS | 60 s | 6,480 / 6,480 | 1,620 | 60 | 182.114 ms | 224.327 ms | 84.578 ms |

The two sustained runs used 360 PPE batch-4 calls, 30 primary batch-2 calls,
1,050 primary batch-4 calls, and 30 RT-DETR batch-2 calls each. The
three-camera remainder safely used bounded singleton fallbacks; there were no
admission overloads, model failures, or route failures.

## Runtime contract

The generic default remains uniform so non-Jetson deployments do not inherit
an unmeasured schedule:

```dotenv
SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT=1.0
```

The measured Jetson batch-4 profile defaults to:

```dotenv
SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT=0.70
```

Raw JSON and console evidence is stored on the Jetson under
`/opt/rakshak-lens/model-server-models/experiments/ppe-25-phase-sweep/`.

## Live rollout

Commit `c2151ae` was pushed directly to `master`, and its hash-matched edge
image was promoted with `SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT=0.70` in
the actual container environment. The warmed model-server container was not
restarted, and the previous edge container remains stopped as rollback.

After the reconnect cycle, edge health was `ok` with no reasons. Cam1 and cam2
were both running, frame-fresh, and using hardware `gstreamer_nvdec`. They had
232 inference successes at the observation point with zero inference failures
or overload drops. Primary, PPE, and RT-DETR transport had zero admission or
route failures, the alert persistence and delivery workers were alive with no
failures, and the model watchdog timer was active.
