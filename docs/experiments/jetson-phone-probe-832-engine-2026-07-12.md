# Jetson 832px phone-probe engine experiment — 2026-07-12

> Superseded later the same day by the calibrated 640px INT8 Small result in
> `jetson-small-primary-filter-rfdetr-2026-07-12.md`. The 832px result remains
> useful evidence that an isolated engine speedup does not establish a camera
> tier, but the production recommendation is now to retire the larger phone
> probe entirely.

## Decision

Keep the deployed 960px YOLO26 Small TensorRT engine for the contextual mobile
phone probe. Reject the 832px candidate: it was faster in isolated inference
but produced a worse service-time tail and did not make twelve cameras clean
under the full mixed workload.

The supported conditional limit therefore remains eleven cameras at 4
detection FPS. The 832 artifact is retained only as rejected Jetson evidence
and is not configured for production.

## Fixed-engine routing finding

The deployed model pack contains fixed 640px and 960px COCO engines. Requests
for 768, 832, 896, and 960 all route to the fixed 960 engine, so merely changing
`mobile_phone_inference_width` inside that range cannot reduce compute. The
four requested widths produced identical confidence values and essentially
identical endpoint latency.

A real intermediate engine was therefore required for a valid experiment.

## Candidate export

YOLO26 Small was exported on the target Jetson as a fixed-shape FP16 832px
TensorRT engine. The low-memory path used:

- an isolated ONNX child process;
- TensorRT 8.5.2.2 `trtexec` heuristic tactics;
- a 256 MiB workspace;
- batch size one and static input shape;
- atomic engine and hash-verified sidecar output.

The TensorRT build completed in 530.251 seconds. It skipped tactics that did
not fit the constrained workspace instead of exhausting unified memory. The
final wrapped engine was 21,335,547 bytes.

## Phone accuracy and isolated latency

Six labelled phone-use positives and four negative controls ran directly
through both fixed engines at confidence 0.10. Production evaluation used a
0.15 phone threshold, 0.30 person context threshold, and the existing
phone-to-person geometry filter.

| Engine | Positive phone-hit images | Raw negative phone-hit images | Actionable negative hits | Person-context images | Median of image medians |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 832 | 4 / 6 | 1 / 4 | 0 / 4 | 8 / 10 | 24.335 ms |
| 960 | 4 / 6 | 1 / 4 | 0 / 4 | 7 / 10 | 29.750 ms |

The one raw negative hit for each engine was `neg-two-phones.jpg`, which had no
production-threshold person. The context and association gates therefore
reject it before alerting. The 832 engine was 18.2% faster in isolation and did
not regress this small engineering corpus.

## Person-crop alternative

A 640px person-crop probe also retained 4/6 positive hits and 0/4 actionable
negative hits after applying the production person threshold and association
geometry. Its median eligible-crop latency was about 27 ms.

It was not promoted because both live cameras also require full-frame person
and animal rules. Replacing one full-frame inference slot with a crop would
silently reduce those capabilities from 4 FPS to 3 FPS. Running both a normal
640px frame and an added crop costs more than the current single 960px probe.
A crop becomes architecturally sound only after a real tracker can carry
full-frame context across the cropped slot, or after a camera profile limits
COCO work to mobile-phone detection alone.

## Exact mixed-load result

The clean candidate run used the production model volume and container name,
semantic model readiness, exact edge admission path, prewarmed per-camera
sessions, raw BGR transport, three admission slots, 4 FPS staggered scheduling,
one phone probe per second, and 11.1% PPE specialist duty.

| Phone engine | Completed | Drops / failures | Minimum FPS | Median | p95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 832 candidate | 1433 / 1440 | 7 / 0 | 3.900 | 21.690 ms | 98.065 ms | 167.546 ms |
| 960 baseline | 1435 / 1440 | 5 / 0 | 3.967 | — | 94.636 ms | 136.665 ms |

The normal 640px primary and PPE work dominate the median, while contention at
the mixed-model tail determines admission loss. The constrained 832 engine's
isolated speedup did not survive that scheduler interaction. It cannot justify
an accuracy-sensitive production change or a new camera-capacity claim.

Several setup-only runs were excluded from the result: two failed before
inference because the candidate omitted the production model-volume mount or
targeted the wrong Docker name, and one was interrupted by an older detached
recovery watchdog. Only the final semantically ready, exact-name, exact-volume
run above is used for the decision.
