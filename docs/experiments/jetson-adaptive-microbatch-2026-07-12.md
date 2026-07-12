# Jetson adaptive batch-2/batch-4 routing (2026-07-12)

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

The promoted batch-4 profile sustains the highest dense-camera throughput, but
an exact two-camera group previously waited the full 10 ms rendezvous window
and then repeated the work through two single-frame requests. This experiment
tests an early batch-2 flush while retaining batch-4 as the maximum route.

## Decision

Promote a 6 ms early flush for exactly two compatible primary or
primary-plus-PPE frames when both the batch-2 and batch-4 TensorRT routes are
ready. Four compatible arrivals still dispatch immediately through batch-4.
One unmatched frame and a three-frame partial group retain the existing 10 ms
fallback behavior; this deliberately avoids splitting a slightly delayed
four-camera group and reducing dense throughput.

The feature is opt-in. A missing batch-2 route disables the early flush and
preserves the existing fallback path.

## Jetson results

The candidate edge module ran inside the live edge image against the live
model server. The benchmark used 360x640 raw frames, a 10 ms maximum rendezvous
window, a 6 ms early flush, three warmups, and 40 measured groups. Cam2's live
workload remained active, making this a stricter shared-device measurement.

| Scenario | Aggregate FPS | Median | p95 | Batch route | Timeout fallbacks |
| --- | ---: | ---: | ---: | --- | ---: |
| Two primary, static batch-4 | 48.495 | 41.004 ms | 43.497 ms | two singleton fallbacks | 86/86 including warmups |
| Two primary, adaptive | 63.809 | 30.735 ms | 33.148 ms | batch-2 | 0/86 |
| Two primary+PPE, adaptive | 38.940 | 50.587 ms | 56.317 ms | specialist batch-2 | 0/86 |

For two primary frames, adaptive routing improved aggregate throughput by
31.6%, reduced median latency by 25.0%, and reduced p95 by 23.8%.

Dense four-frame behavior was measured separately with five warmups and 100
groups:

| Four-primary route | Aggregate FPS | Median | p95 | Batch-4 groups | Timeout frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 101.193 | 37.998 ms | 45.947 ms | 103/105 | 8/420 |
| Adaptive | 101.938 | 38.041 ms | 44.357 ms | 103/105 | 8/420 |

The identical two scheduler-jitter misses with the feature both disabled and
enabled show that the early flush did not create the misses. The sustained
throughput difference is within run-to-run noise. Production's deterministic
four-camera phase scheduler previously completed the 20-camera, 60-second cold
gate with zero fallbacks; that remains the capacity contract.

No tested adaptive route produced an admission overload or model-server route
fallback. Local regression coverage completed 217 focused scheduler, capture,
transport, TensorRT, grouped-inference, result, and video-processing tests.

## Runtime contract

Set these together only after both fixed-batch engine pairs are installed:

```dotenv
SAFETYLENS_REMOTE_FRAME_BATCH_SIZE=4
SAFETYLENS_REMOTE_PRIMARY_BATCH_WAIT_SECONDS=0.010
SAFETYLENS_REMOTE_SPECIALIST_BATCH_WAIT_SECONDS=0.010
SAFETYLENS_REMOTE_BATCH2_EARLY_FLUSH_SECONDS=0.006
SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE=4
```

`scripts/benchmark_adaptive_microbatch_jetson.py` reproduces the two- and
four-frame route comparisons without printing model-server credentials.
