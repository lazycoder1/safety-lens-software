# Jetson immutable inference-frame experiment — 2026-07-12

## Decision

Retain the captured NumPy frame for asynchronous inference instead of copying
its full pixel buffer at submission. Mark the shared array read-only first so
an accidental in-place write fails rather than racing inference, streaming, or
alert evidence.

The camera loop replaces its local `frame` variable on the next capture. The
inference `Future` retains the prior array reference, so its buffer remains
alive until detection and alert processing complete.

## Copy microbenchmark

The exact Jetson edge runtime measured 2000 operations per repeat and retained
the best of five repeats:

| Frame | Buffer | `frame.copy()` | Read-only view | Construction speedup |
| --- | ---: | ---: | ---: | ---: |
| 352x288 BGR | 304,128 bytes | 24.737 µs | 0.407 µs | 60.8x |
| 640x360 BGR | 691,200 bytes | 51.168 µs | 0.410 µs | 124.9x |
| 960x540 BGR | 1,555,200 bytes | 108.103 µs | 0.411 µs | 262.8x |

The primary benefit is removing one 0.3–1.6 MB allocation per submitted camera
frame. Combined with raw-body memoryview transport, the edge no longer makes
either of the two previous full-pixel copies before same-host model inference.

## Mutation audit

The concurrent paths consume the captured frame read-only:

- motion and stream signatures use OpenCV read operations;
- stream resize/JPEG encoding reads the clean frame;
- detection and pose overlays copy before drawing;
- remote inference resizes or creates a contiguous view without mutation;
- alert evidence JPEG encoding reads the retained inference frame.

A focused test proves the submitted object is the same array, has its writeable
flag disabled, and rejects item assignment.

## Live validation

The candidate ran both production RTSP cameras through GStreamer/NVDEC and
forced active MJPEG subscribers for five seconds per camera:

| Camera | MJPEG bytes received | Frame fresh | Inference drops / failures | Capture failures |
| --- | ---: | --- | ---: | ---: |
| cam1 | 296,588 | yes | 0 / 0 | 0 |
| cam2 | 354,418 | yes | 0 / 0 | 0 |

No read-only/writeable exception or mutation log was emitted. Alert
persistence, callback, delivery, and backpressure failures remained zero.

The harsher ten-virtual-plus-two-live exact-edge stress run completed 1194 of
1200 jobs, with six bounded overload drops, zero failures, 21.828 ms median,
86.170 ms p95, and 288.346 ms maximum. This overlaps the copied-frame baseline
and does not establish a new capacity tier.

Validation: 87 focused video-processing, rendering, stream-throttling,
fresh-alert-vote, and grouped-inference tests passed.
