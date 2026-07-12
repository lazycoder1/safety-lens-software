# Jetson concurrent specialist-batch experiment — 2026-07-12

## Decision

Promote opt-in concurrent execution of the independent YOLO26 Small primary
batch-2 and YOLOE-26S PPE batch-2 engines on the tested Orin NX. The existing
four-worker model executor bounds the overlap; generic deployments remain
sequential unless `SAFETYLENS_SPECIALIST_BATCH_CONCURRENT=true` is explicitly
set after a device-specific load and accuracy gate.

This moves the measured conditional inference tier from sixteen to eighteen
camera-equivalents at 4 FPS. Nineteen is rejected because its 60-second cold
repeat dropped eight primary requests and exceeded the 250 ms frame period.
Twenty is also rejected because its 30-second run dropped two specialist
requests (one paired execution).

This remains a Small-only result. Nano is not used.

## Isolated engine overlap

The benchmark alternated sequential and concurrent execution over 200 timed
frame pairs drawn from all 18 PPE validation images. Each iteration used the
same fixed YOLO26 Small INT8 batch-2 engine and fixed-prompt YOLOE-26S FP16
batch-2 engine.

| Execution | Mean | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Sequential primary then PPE | 53.602 ms | 53.169 ms | 55.730 ms | 56.242 ms |
| Concurrent primary and PPE | 43.267 ms | 43.142 ms | 45.534 ms | 49.840 ms |

Concurrent execution reduced median pair latency by 23.2%. All 200 paired
results matched exactly between execution modes, including class IDs,
confidences, boxes, and prompt-synonym deduplication output.

The reusable harness is
`scripts/benchmark_specialist_batch_overlap_jetson.py`. Raw evidence is stored
on the Jetson under
`/opt/rakshak-lens/model-server-models/experiments/specialist-overlap/`.

## Exact edge-load result

The load gate exercised the real edge `predict_record_batches()` path with:

- 4 FPS per camera;
- YOLO26 Small INT8 primary inference at 640px;
- 11.1% conservative PPE specialist duty;
- paired camera phases;
- 6 ms primary and specialist rendezvous windows;
- four bounded remote admission slots and a 125 ms admission wait;
- phone and PPE burst overlap avoidance.

| Cameras | Duration | Completed | Drops / failures | Specialists | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 30 s | 1,920 / 1,920 | 0 / 0 | 208 / 208 | 103.841 ms | 183.818 ms | pass |
| 17 | 30 s | 2,040 / 2,040 | 0 / 0 | 221 / 221 | 118.615 ms | 191.218 ms | pass |
| 18 | 30 s | 2,160 / 2,160 | 0 / 0 | 234 / 234 | 131.532 ms | 184.557 ms | pass |
| 19 | 30 s | 2,280 / 2,280 | 0 / 0 | 247 / 247 | 154.760 ms | 219.428 ms | provisional only |
| 20 | 30 s | 2,398 / 2,400 | 2 / 0 | 258 completed | 170.677 ms | 222.008 ms | reject: drops |
| 19 | 60 s cold repeat | 4,552 / 4,560 | 8 / 0 | 494 / 494 | 155.619 ms | 257.697 ms | reject: drops and stale tail |
| 18 | 60 s cold repeat | 4,320 / 4,320 | 0 / 0 | 468 / 468 | 132.783 ms | 222.702 ms | promote |

The eighteen-camera cold repeat paired all 3,852 primary requests and all 468
specialist requests without a timeout fallback. Every camera completed exactly
4 FPS and every scheduled specialist evaluation was preserved.

## Comparison with the sequential production tier

At sixteen cameras, the prior sequential specialist route recorded p95
118.532 ms and maximum 210.856 ms over its 60-second promotion run. The
concurrent 30-second comparison recorded p95 103.841 ms and maximum 183.818 ms.
The exact runs differ in duration, so these latency deltas are supporting
evidence; the new 60-second eighteen-camera cold repeat is the authoritative
capacity gate.

## Operational constraints

- The flag is disabled by default because concurrent TensorRT contexts increase
  instantaneous GPU pressure and may regress smaller-memory or differently
  scheduled devices.
- The endpoint retains the existing HTTP 409 fallback when either batch-2
  runtime is unavailable. Edge requests then reuse the proven grouped
  single-frame path.
- The existing executor limits concurrent model tasks to four; the change does
  not create an unbounded thread pool.
- Eighteen is valid for the measured 11.1% conditional PPE workload, not for
  always-on PPE across every camera.
- Real RTSP/NVDEC ingest is still validated only to four simultaneous streams
  at the office site. Eighteen is inference capacity, not an eighteen-decoder
  certification.

## Admission-concurrency follow-up

The concurrent model scheduler changed the queue shape, so edge admission was
re-swept before attempting a larger TensorRT batch. Increasing admission slots
did not produce a repeatable freshness-safe tier above eighteen.

| Cameras | Slots / wait | Duration | Completed | Drops | p95 | Maximum | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 19 | 5 / 125 ms | 30 s | 2,278 / 2,280 | 2 | 156.226 ms | 234.233 ms | reject: drops |
| 20 | 5 / 125 ms | 30 s | 2,400 / 2,400 | 0 | 167.060 ms | 235.707 ms | provisional only |
| 22 | 5 / 125 ms | 30 s | 2,634 / 2,640 | 6 | 207.502 ms | 267.954 ms | reject: drops and stale tail |
| 20 | 5 / 125 ms | 60 s cold repeat | 4,800 / 4,800 | 0 | 172.434 ms | 261.926 ms | reject: stale tail |
| 20 | 5 / 100 ms | 30 s | 2,400 / 2,400 | 0 | 166.991 ms | 247.739 ms | provisional only |
| 20 | 5 / 100 ms | 60 s cold repeat | 4,798 / 4,800 | 2 | 172.373 ms | 272.980 ms | reject: drops and stale tail |
| 20 | 6 / 75 ms | 30 s | 2,400 / 2,400 | 0 | 164.450 ms | 277.768 ms | reject: stale tail |

The five-slot 125 ms run demonstrates why request completion alone is not a
promotion gate: it retained every frame by allowing one request to finish
11.926 ms after the next frame period began. Shortening the admission wait made
the short run appear clean, but the cold repeat still dropped a primary pair.
Six slots increased instantaneous contention and worsened the tail despite
completing every short-run request.

Production therefore remains at four admission slots, a 125 ms bounded wait,
and the eighteen-camera supported tier. Raising queue depth would hide overload
as stale inference rather than create capacity. A higher tier requires more
aggregate engine throughput, such as a separately accuracy-gated larger
microbatch, rather than another admission-limit increase.
