# Jetson partial-group admission smoothing — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

Promote a 14 ms primary and specialist rendezvous window for the validated
batch-4 Orin NX profile. Full four-frame groups still dispatch immediately,
and exact two-frame remainders still use the existing 6 ms batch-2 early
flush. The longer window therefore changes only singleton and three-frame
partial groups.

This raises the supported conditional-inference tier from 20 to 21 camera
equivalents at 4 AI FPS per camera with 11.1% PPE specialist duty. Two
independent 60-second runs completed 5,040/5,040 requests with zero overloads
or failures and maximum latency below the next 250 ms inference period.
Twenty-two cameras is rejected because its 255.802 ms maximum crossed the
freshness limit.

This remains a Small-only result. YOLO26 Nano is not used.

## Bottleneck

The batch-4 profile divides 21 cameras into five full four-camera phases and
one singleton phase. The singleton cannot form a fixed TensorRT batch and uses
the existing single-frame fallback after the bounded rendezvous expires.

The prior 10 ms profile completed all 5,040 requests in its 21-camera cold
gate but peaked at 258.433 ms, 8.433 ms beyond the next inference period. The
problem was queue shape and tail freshness, not missing compute or an
unbounded queue.

## Rejected direct bypass

A phase-aware candidate marked the known remainder as a singleton so it could
skip the 10 ms rendezvous. It behaved exactly as implemented: 107 primary and
13 specialist timeout fallbacks became direct singleton bypasses in the
30-second run. However, dispatching immediately caused the singleton to
contend with the preceding full batch.

| 21-camera shape | p95 | Maximum | Primary / specialist remainder outcome |
| --- | ---: | ---: | --- |
| 10 ms control | 175.465 ms | 234.822 ms | 107 / 13 timeout fallbacks |
| Direct singleton bypass | 184.690 ms | 256.395 ms | 107 / 13 direct bypasses |

The bypass worsened p95 by 9.225 ms and maximum by 21.573 ms. Its runtime code
was removed. The benchmark retains `--phase-remainder-hint` so the negative
result is reproducible.

## Wait sweep

All short runs completed 2,520/2,520 requests at exactly 4 FPS per camera with
zero overloads and failures. Full batch-4 calls remained immediate; only the
singleton remainder observed the configured maximum wait.

| Partial-group wait | Median | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | --- |
| 10 ms | 40.749 ms | 175.465 ms | 234.822 ms | control |
| 12 ms | 40.671 ms | 175.126 ms | 233.759 ms | improved |
| 14 ms | 40.625 ms | 173.407 ms | 229.631 ms | select |
| 16 ms | 40.782 ms | 168.645 ms | 234.209 ms | lower p95, worse maximum |

The result is intentionally non-monotonic. A slightly longer delay lets the
preceding full batch release GPU/admission pressure, but excessive waiting
spends the singleton's remaining freshness budget.

## Promotion and boundary gates

The exact edge workload used:

- 640-pixel YOLO26 Small INT8 primary inference;
- 4 FPS per camera and four-camera phases;
- 11.1% YOLOE-26S FP16 PPE specialist duty;
- 6 ms batch-2 early flush;
- four bounded admission slots and a 125 ms admission wait;
- deferred phone/PPE overlap and raw frame transport.

| Cameras | Duration | Completed | Drops / failures | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 30 s | 2,400 / 2,400 | 0 / 0 | 148.182 ms | 184.661 ms | regression pass |
| 21 | 60 s A | 5,040 / 5,040 | 0 / 0 | 171.523 ms | 242.346 ms | pass |
| 21 | 60 s B | 5,040 / 5,040 | 0 / 0 | 172.484 ms | 237.275 ms | promote |
| 22 | 30 s | 2,640 / 2,640 | 0 / 0 | 189.804 ms | 255.802 ms | reject: stale tail |

At 21 cameras, each cold run preserved all 4,280 batch-4 primary frames and
520 batch-4 specialist frames. The 214 primary and 26 specialist singleton
remainders used the bounded fallback with no route fallbacks or admission
overloads. At 22 cameras, the two-camera remainder used the 6 ms batch-2 route,
so increasing the larger wait did not hide the next capacity boundary.

## Live deployment

Only these measured profile values changed:

```dotenv
SAFETYLENS_REMOTE_PRIMARY_BATCH_WAIT_SECONDS=0.014
SAFETYLENS_REMOTE_SPECIALIST_BATCH_WAIT_SECONDS=0.014
```

The previous 10 ms edge container remains stopped as the immediate rollback.
The promoted container retained NVIDIA runtime, mounts, media devices, and
the existing Small-model/batch-4 engine contract. During a 30-second live
soak, cam2 stayed fresh on `gstreamer_nvdec`, inference successes rose from 26
to 62, and overload/failure counters remained zero. Cam1 remained in its
pre-existing source outage.

Raw evidence is stored on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/phase-remainder/`
