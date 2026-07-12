# Jetson single-process edge capacity experiment — 2026-07-12

## Decision

Support eleven conditional cameras at 4 detection FPS on the tested Jetson
with YOLO26 Small. Do not advertise twelve as a clean tier: after removing
control-plane and connection cold starts, twelve still shed five of 1440
scheduled jobs.

Prewarm each camera inference thread's model-server health connection before
its first bounded inference request. The warmup is best effort and happens
outside GPU admission, so a temporary model-server outage still falls through
to the normal retry path without occupying one of the three Jetson inference
slots.

## Corrected method

Earlier `edge` stress runs started a second client process while the production
edge process remained active. Those processes had independent admission
semaphores, so that shape was useful as a saturation test but could not prove
the production capacity boundary.

For this experiment the production edge process was stopped for each 30-second
run. One standalone process then exercised the exact
`model_manager.predict_record_batches()` path, including the production remote
admission semaphore, raw BGR transport, and thread-local HTTP sessions. The
model server remained unchanged.

Each virtual camera requested:

- YOLO26 Small primary inference at 4 FPS and 640px;
- a 960px phone probe once per second when person context was present;
- the Small PPE specialist at 11.1% deterministic duty;
- the production staggered camera phase and three-slot Jetson admission.

Four representative office frames were replayed across the virtual cameras.
Ingest capacity was measured separately and was not included in this compute
test.

## Cold-start finding

The first version began timed traffic before loading the remote configuration
and before each worker established its thread-local HTTP connection. Most of
the apparent low-camera loss was therefore startup work rather than sustained
GPU pressure:

| Cameras | Cold completion | Cold drops | Config-warm completion | Config-warm drops |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1194 / 1200 | 6 | 1198 / 1200 | 2 |
| 11 | 1312 / 1320 | 8 | 1318 / 1320 | 2 |
| 12 | 1428 / 1440 | 12 | 1433 / 1440 | 7 |

Prewarming both the control plane and every worker's thread-local keep-alive
socket removed that artifact from the timed window.

## Final steady-state boundary

| Cameras | Completed | Drops / failures | Minimum FPS | Median | p95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 1320 / 1320 | 0 / 0 | 4.000 | 20.954 ms | 79.471 ms | 120.165 ms |
| 12 | 1435 / 1440 | 5 / 0 | 3.967 | — | 94.636 ms | 136.665 ms |

Eleven is the demonstrated zero-drop conditional compute boundary. Twelve
completed 99.65% of scheduled work, but bounded overload shedding proves it is
above the clean supported tier for this workload.

The conservative all-person planning limit remains ten cameras because every
camera then activates the high-resolution phone-probe path. That heavier mix
was established in the prior admission experiment and was not promoted by this
connection-warmup change.

## Admission-wait follow-up

The exact-edge benchmark exposed `--admission-timeout`, but edge mode initially
left the production module's 65 ms constant unchanged. The harness now applies
the requested timeout to the exact model-manager admission path before starting
workers.

Twelve cameras were retested with the 960px phone probe active once per second
and 11.1% PPE specialist duty:

| Admission wait | Completed | Drops / failures | Minimum FPS | Median | p95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 ms | 1435 / 1440 | 5 / 0 | 3.967 | — | 94.636 ms | 136.665 ms |
| 80 ms | 1431 / 1440 | 9 / 0 | 3.900 | 30.982 ms | 114.298 ms | 176.797 ms |
| 100 ms | 1433 / 1440 | 7 / 0 | 3.900 | 32.529 ms | 114.222 ms | 199.303 ms |

Longer waits convert contention into queue time without clearing the
twelve-camera tier. Both candidates raised median and tail latency, and neither
completed every scheduled job. Keep the 65 ms production wait. A clean twelfth
camera requires less service work or real cross-camera batching, not a deeper
client queue.

## Ingest versus inference

Twelve simultaneous RTSP/NVDEC pipelines have separately sustained 8 decoded
FPS while replaying two office streams. That proves decoder capacity, not full
analytics capacity. The product limit is the lower compute result:

- eleven cameras for the measured conditional workload;
- ten cameras for conservative all-person planning;
- twelve ingest pipelines only, without a claim of clean 4 FPS analytics.

## Live validation

The session-prewarm candidate started both production RTSP cameras on
GStreamer/NVDEC. Both reported fresh frames, successful inference, zero
inference overload drops or failures, and zero capture failures. No prewarm
failure or executor-initializer exception was logged.
