# Jetson phone-probe context gating — 2026-07-12

## Decision

Run the 960px mobile-phone recall probe only after the normal 640px YOLO26
Small primary pass detects a person. The 640px primary remains active for every
scheduled inference, so a newly arriving person can enable the next 960px probe
immediately.

The gate accepts only `coco_primary` person context. A person emitted by a PPE
or other specialist cannot recursively enable the phone probe.

## Accuracy gate

Six phone-use positives and four negatives were each resized to 854x480 and
352x288. Every sample ran through the live Jetson raw model-server path at both
640px and 960px.

| Evidence | Result |
| --- | ---: |
| Positive source variants | 12 |
| Variants with a 640px person | 12 / 12 |
| 960px alert-quality phone/person hits | 8 |
| Hits preserved by the 640px person gate | 8 / 8 |

The scoped set therefore showed no recall loss from the gate. This does not
replace a customer acceptance dataset; it proves that the existing 960px hits
in the engineering set all have cheap-primary person context.

## Live waste baseline

Both production cameras had no current person detections during the baseline
window, but each still scheduled about 27 high-resolution phone probes in 30
seconds. The recent 30-sample histories contained 32 probe samples combined and
no current person sample. Those probes produced zero phone hits.

## Capacity result

The load shape kept two RTSP cameras live and replayed virtual office cameras at
4 FPS. Every virtual camera ran 640px Small COCO, used 11.1% Small PPE duty, and
had a 960px phone-probe slot once per second. Raw transport and the 65ms bounded
admission wait remained enabled.

| Total camera-equivalents | Person-context duty | Completed | Drops | Minimum FPS | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 100% | 1074 / 1080 | 6 | 3.933 | 64.998 ms |
| 11 | 25% | 1080 / 1080 | 0 | 4.000 | 50.552 ms |
| 12 | 25% | 1199 / 1200 | 1 | 3.967 | 56.091 ms |

The zero-drop conditional compute boundary is eleven camera-equivalents at 4
FPS when person context occupies 25% of phone-probe slots. The all-person worst
case remains ten. Four simultaneous live RTSP/NVDEC pipelines are still the
separate ingest validation limit.

## Observability

`phoneProbe.contextSuppressedCount`, `lastContextSuppressedAt`, the suppression
reason, and its health age distinguish a healthy context gate from a stalled
phone detector. Suppression telemetry is throttled to the configured probe
interval, while person discovery is checked on every normal inference result.
