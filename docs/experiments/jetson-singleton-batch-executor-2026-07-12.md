# Jetson singleton batch-executor experiment — 2026-07-12

## Decision

Execute a one-model grouped request directly in the FastAPI worker instead of
submitting it to the model server's nested four-worker batch executor and then
blocking on its `Future`.

True multi-model batches still use the executor so COCO and PPE runtimes can
overlap. Response structure, model arguments, detection records, and edge
admission are unchanged.

## Root cause

After contextual specialist gating, most scheduled requests contain only the
YOLO26 Small primary. `_run_inference_batch()` nevertheless queued every
singleton into `_BATCH_INFERENCE_EXECUTOR`, allocated a `Future`, and
immediately waited for it. FastAPI already runs the synchronous route in a
worker thread, so the second scheduling boundary did not provide concurrency.

## A/B result

Both production cameras remained live. Each isolated primary-only run replayed
ten virtual cameras at 4 FPS for 30 seconds through raw transport with
three-slot admission: 1200 grouped endpoint requests per run.

| Model-server path | Run | Completed | Drops / failures | Median | p95 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Nested executor baseline | A | 1200 / 1200 | 0 / 0 | 21.529 ms | 40.686 ms | 67.320 ms |
| Nested executor baseline | B | 1200 / 1200 | 0 / 0 | 21.464 ms | 39.624 ms | 58.176 ms |
| Direct singleton | A | 1200 / 1200 | 0 / 0 | 21.091 ms | 38.892 ms | 67.335 ms |
| Direct singleton | B | 1200 / 1200 | 0 / 0 | 21.267 ms | 39.450 ms | 58.364 ms |

The two-run averages improve by approximately 1.5% at median and 2.5% at p95.
The result is intentionally reported as a latency cleanup, not a new camera
capacity tier.

Two conditional 12-camera-equivalent candidate runs also completed all 1200
jobs at 4 FPS with zero drops or failures. Their p95 values were 65.643 and
75.909 ms, compared with 68.071 and 77.271 ms in the preceding baseline runs.

## Validation

- 54 focused remote-transport and grouped-inference tests passed.
- A unit guard fails if a singleton batch reaches the nested executor.
- Multi-model batches retain the existing parallel executor path.
- After the controlled model-server restarts, a clean production soak added
  133 cam2 and 32 cam1 inference successes with no new inference failures or
  overload drops.
- Both cameras remained frame-fresh; capture and alert-pipeline failures stayed
  at zero.
