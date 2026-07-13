# Jetson tracked PPE substitution — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

Promote an explicit 0.5 FPS PPE cadence and optional tracked-context
substitution. For PPE-enabled cameras using the deployed fixed rider-helmet
prompt contract, the repeatably freshness-safe inference tier is 24 camera
equivalents at four scheduled decision slots per second:

- 3.5 fresh YOLO26 Small primary observations per camera per second;
- 0.5 fresh YOLOE PPE observations per camera per second;
- cached primary person/vehicle context is used only for the PPE association;
- unrelated alert state is not advanced or cleared by a partial PPE frame.

Twenty-five cameras is rejected. It completed every request but one of two
60-second runs reached 282.405 ms PPE latency, above the 250 ms camera period.
The existing non-PPE tier remains 21 cameras at four full primary FPS. The
RT-DETR phone tier remains 20 cameras at four effective decision FPS plus one
device-wide RT-DETR frame per second. These are separate measured profiles;
their maxima must not be combined arithmetically.

## Problem found

Previous capacity tests budgeted the PPE specialist at about 11.1% duty, but
the camera worker had no explicit PPE cadence. After primary context became
actionable, a persistent person could cause PPE inference on every four-FPS
primary pass. The benchmark and the live execution contract therefore did not
match.

This was both a capacity risk and an engineering observability problem: the
device could appear healthy in a sparse scene, then multiply specialist work
when people entered PPE cameras.

## Implementation

The worker now applies a monotonic, per-camera PPE cadence before submitting
inference. The default target is 0.5 FPS and applies even while substitution is
disabled, so the configured duty is a real runtime contract rather than a
benchmark-only assumption.

When substitution is enabled, a due PPE frame replaces its normal primary
frame only when all of these conditions hold:

1. the camera plan requires both COCO primary and PPE specialist inference;
2. the lightweight primary person tracker has a stable, unexpired track;
3. cached full-primary context is still within the existing one-second TTL;
4. no unvalidated companion specialist requires the same frame;
5. the cached context contains only person and rider-vehicle classes needed
   for PPE association.

An unstable or expired track keeps the due pass additive. A non-due frame runs
the primary normally and suppresses only PPE. RT-DETR phone substitution is
scheduled first; a selected phone pass defers PPE rather than silently losing
it.

The model server adds authenticated raw-frame PPE-only batch-2 and batch-4
routes. The edge reuses the existing bounded microbatch implementation for a
single configured model key, rather than introducing a second batching stack.
Prompt mismatch returns to the existing generic route without globally
poisoning route discovery.

PPE substitution is an explicitly partial observation. Only PPE capabilities
advance; vehicle, animal, crowd, zone, object-lifecycle, fall, phone, and other
rule windows remain unchanged. This prevents a PPE-only frame from falsely
clearing an unrelated incident.

## Worker-path semantic gate

All 18 existing PPE validation images were replayed through both paths:

1. additive primary plus PPE inference;
2. primary-only context followed by PPE-only inference and cached-context
   composition.

The comparison used the actual worker normalization, batch transport, fixed
TensorRT runtimes, person/vehicle association, and violation function.

| Gate | Result |
| --- | ---: |
| Relevant detection parity | 18 / 18 frames |
| Violation outcome parity | 18 / 18 frames |
| PPE batch-4 transport | 16 frames / 4 calls |
| PPE batch-2 transport | 2 frames / 1 call |
| Route fallbacks / overloads / timeouts | 0 / 0 / 0 |

The corpus contains no actionable rider-helmet violation after primary rider
association, so alert-positive and unrelated-state behavior is additionally
covered by the local synthetic geometry and partial-observation tests. The
Jetson gate proves real detector-output parity, not broad ground-truth PPE
accuracy beyond the existing corpus.

## Exact 0.5 FPS capacity gate

The load used 640-pixel YOLO26 Small INT8 and YOLOE-26S FP16, four-camera
phases, fixed batch-4 with batch-2 remainder handling, four bounded admission
slots, 125 ms admission, 14 ms batch wait, and 12.5% PPE duty at four scheduled
decision FPS. PPE work replaced a primary slot and used the new PPE-only edge
transport.

| Cameras | Duration | Effective decisions | PPE passes | Drops / failures | Primary maximum | PPE maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 28 | 30 s | 3,360 / 3,360 | 420 | 0 / 0 | 224.908 ms | 293.937 ms | reject PPE tail |
| 26 | 30 s | 3,120 / 3,120 | 390 | 0 / 0 | 212.795 ms | 276.841 ms | reject PPE tail |
| 25 | 30 s | 3,000 / 3,000 | 375 | 0 / 0 | 101.006 ms | 194.231 ms | provisional |
| 25, run 1 | 60 s | 6,000 / 6,000 | 750 | 0 / 0 | 205.439 ms | 282.405 ms | reject tail |
| 25, run 2 | 60 s | 6,000 / 6,000 | 750 | 0 / 0 | 188.703 ms | 236.614 ms | pass only |
| 24 | 30 s | 2,880 / 2,880 | 360 | 0 / 0 | 93.230 ms | 149.786 ms | pass |
| 24, run 1 | 60 s | 5,760 / 5,760 | 720 | 0 / 0 | 135.679 ms | 219.121 ms | pass |
| 24, run 2 | 60 s | 5,760 / 5,760 | 720 | 0 / 0 | 120.590 ms | 169.144 ms | pass |

Both sustained 24-camera runs used exactly 180 PPE batch-4 calls with no
partial batch, singleton, timeout, route-fallback, admission, or model failure.

## Alert-latency and scope notes

The 0.5 FPS specialist cadence means five persistent rider-helmet votes take
about ten seconds. A generic PPE rule using the fallback ten-vote threshold
takes about twenty seconds unless its rule has an explicit lower threshold.
This trades alert confirmation speed for deterministic capacity and false-alert
resistance; it must remain visible in camera/rule configuration rather than be
described as four fresh PPE observations per second.

The 24-camera capacity result uses the deployed fixed rider-helmet prompt set.
Other dynamic prompt sets safely fall back to the generic PPE transport but do
not inherit this measured tier until separately load-gated. Likewise, the
24-camera PPE tier has not yet been combined with the one-FPS device-wide
RT-DETR phone budget.

## Validation and evidence

- 368 relevant local tests passed;
- Ruff passed on every changed runtime, verifier, and test file;
- `git diff --check` passed;
- candidate image runtime-source hashes matched the local backend sources;
- worker parity and raw load evidence is stored on the Jetson under
  `/opt/rakshak-lens/model-server-models/experiments/ppe-substitution/`.

The checked-in substitution default remains off for conservative upgrades.

## Live promotion

Commit `fd79647` was pushed directly to `master`. The hash-matched edge and
model-server candidate images were then promoted on the Jetson, while the prior
containers were retained as stopped rollback artifacts. The device config now
sets `ppe_specialist_target_fps` to `0.5` and explicitly enables substitution.

After promotion:

- the model server reported all primary/PPE batch-2 and batch-4 engines warmed,
  plus both RT-DETR phone engines warmed;
- the edge health response exposed primary, PPE-only, specialist, RT-DETR, and
  substitution counters with zero route or admission failures;
- cam2 remained fresh on `gstreamer_nvdec` with zero inference overloads or
  failures;
- cam1 remained in its pre-existing source outage and CPU fallback state;
- the model watchdog timer was active.

Cam2 has the rider-helmet capability, but the observed scene had no current
rider-vehicle context. The context gate correctly emitted no PPE work instead
of spending specialist inference on an unactionable empty scene. Existing
phone-person context did exercise seven RT-DETR batch-1 frames without failure,
showing that the previously promoted phone route remained intact after the
container swap.

## Combined PPE plus RT-DETR capacity follow-up

Cam2 requires both rider-helmet and mobile-phone capabilities, so the PPE and
RT-DETR maxima cannot be treated as independent deployment choices. The load
harness now accepts a deterministic substitution sequence offset. At one-eighth
duty, an offset of one schedules the two selected RT-DETR frames one camera
period before their PPE frames. This models the live worker order, where an RT
selection defers an otherwise due PPE pass to the next primary slot, without
allowing the two specialists to collapse into one benchmark event.

The combined workload kept PPE substitution at 0.5 FPS on every camera and
added exactly one aggregate RT-DETR phone frame per second across the device.
The two RT-qualified cameras therefore received 3.0 full primary, 0.5 PPE, and
0.5 RT observations per second. Other PPE cameras received 3.5 primary and 0.5
PPE observations per second.

| Cameras | Duration | Effective decisions | PPE frames | RT frames | Drops / failures | Primary maximum | PPE maximum | RT maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 30 s | 2,880 / 2,880 | 360 | 30 | 0 / 0 | 124.236 ms | 246.475 ms | 104.135 ms | provisional |
| 23 | 30 s | 2,760 / 2,760 | 345 | 30 | 0 / 0 | 139.676 ms | 219.598 ms | 144.282 ms | pass |
| 22 | 30 s | 2,640 / 2,640 | 330 | 30 | 0 / 0 | 69.873 ms | 142.136 ms | 73.923 ms | pass |
| 24, run 1 | 60 s | 5,760 / 5,760 | 720 | 60 | 0 / 0 | 91.062 ms | 158.819 ms | 105.026 ms | pass |
| 24, run 2 | 60 s | 5,760 / 5,760 | 720 | 60 | 0 / 0 | 92.889 ms | 161.364 ms | 108.593 ms | pass |

Every sustained 24-camera request used the intended PPE batch-4 or RT-DETR
batch-2 route, with no singleton, timeout, route-fallback, admission, or model
failure. Twenty-five cameras was not repeated with the extra RT load because
the strictly cheaper PPE-only profile had already violated the 250 ms limit.
Therefore 24 cameras is also the repeatably safe combined maximum for the
deployed rider-helmet plus one-FPS device-wide phone-recall profile.

Raw combined evidence is stored under
`/opt/rakshak-lens/model-server-models/experiments/ppe-rtdetr-combined/`.
After the sweep, cam2 returned fresh on `gstreamer_nvdec`, the edge and model
server reported zero inference failures or overloads, and the watchdog was
active. Cam1 remained in its pre-existing source outage.

## Confirmation-time cadence acceleration

The 0.5 FPS scout cadence bounds idle and non-violating PPE load, but a
five-vote rider-helmet rule otherwise needs about ten seconds after its first
observation. The worker now supports a separate
`ppe_specialist_confirmation_fps`, defaulting to 1.0 FPS. It accelerates only
while a `Missing ...` rule has a pending positive vote or is active. Animal,
phone, vehicle, zone, and other incident state cannot trigger this faster PPE
cadence.

At the configured rates, a rider violation requires at most about six seconds
to confirm: up to two seconds for the 0.5 FPS scout to find the first violation,
then four more votes at one-second intervals. A generic ten-vote PPE rule has
an approximately eleven-second upper bound instead of twenty seconds. The
confirmation rate is clamped to never be slower than the scout rate.

Before changing the worker, the worst case was load-gated: all 24 cameras ran
PPE at the one-FPS confirmation rate simultaneously, while the device retained
one aggregate RT-DETR phone FPS. This is deliberately stricter than the normal
case where only cameras with pending PPE incidents accelerate.

| Cameras | Duration | Effective decisions | PPE frames | RT frames | Drops / failures | Primary maximum | PPE maximum | RT maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24 | 30 s | 2,880 / 2,880 | 720 | 30 | 0 / 0 | 87.258 ms | 157.558 ms | 105.129 ms | pass |
| 23 | 30 s | 2,760 / 2,760 | 690 | 30 | 0 / 0 | 104.237 ms | 177.830 ms | 108.299 ms | pass |
| 22 | 30 s | 2,640 / 2,640 | 660 | 30 | 0 / 0 | 64.612 ms | 139.079 ms | 59.045 ms | pass |
| 24, run 1 | 60 s | 5,760 / 5,760 | 1,440 | 60 | 0 / 0 | 91.562 ms | 158.951 ms | 104.857 ms | pass |
| 24, run 2 | 60 s | 5,760 / 5,760 | 1,440 | 60 | 0 / 0 | 91.444 ms | 156.211 ms | 104.891 ms | pass |

Both sustained 24-camera runs used exactly 360 PPE batch-4 calls and 30
RT-DETR batch-2 calls. They recorded no singleton, timeout, fallback,
admission, or model failures. Confirmation acceleration therefore improves
alert latency without reducing the established 24-camera tier.

Raw confirmation-load evidence is stored under
`/opt/rakshak-lens/model-server-models/experiments/ppe-confirmation-1fps/`.
