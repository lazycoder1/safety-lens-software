# Jetson raw-frame memoryview experiment — 2026-07-12

## Decision

Send the already contiguous raw BGR inference frame through a byte-cast
`memoryview` instead of materializing `ndarray.tobytes()` for every remote
request.

Requests retains the memoryview as the prepared body, calculates Content-Length
from its byte-cast length, and performs the synchronous send while the backing
NumPy array is alive. The model server endpoint and raw frame format are
unchanged.

## Copy microbenchmark

The exact Jetson edge Python runtime measured 2000 operations per repeat and
retained the best of five repeats:

| Raw frame | Payload | `tobytes()` | Byte memoryview | Construction speedup |
| --- | ---: | ---: | ---: | ---: |
| 640x360 BGR | 691,200 bytes | 50.502 µs | 0.425 µs | 118.8x |
| 960x540 BGR | 1,555,200 bytes | 107.024 µs | 0.430 µs | 249.0x |

The important result is allocation removal: the previous path created a second
0.7–1.6 MB object per admitted job. The byte view is approximately 0.4 µs and
does not duplicate the frame buffer.

## End-to-end stress result

Both production cameras remained live while ten virtual workers used the exact
edge `predict_record_batches()` path at 4 FPS. This intentionally harsher shape
places the virtual and production workers in separate processes, so it is used
for A/B comparison rather than a supported camera-count claim.

| Raw body | Run | Completed | Drops | Failures | Median | p95 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bytes copy | A | 1192 / 1200 | 8 | 0 | 21.980 ms | 74.309 ms | 324.066 ms |
| Bytes copy | B | 1193 / 1200 | 7 | 0 | 21.883 ms | 85.424 ms | 305.754 ms |
| Memoryview | A | 1194 / 1200 | 6 | 0 | 21.946 ms | 82.358 ms | 292.381 ms |
| Memoryview | B | 1193 / 1200 | 7 | 0 | 21.594 ms | 82.251 ms | 276.654 ms |

End-to-end tails overlap and remain dominated by GPU/runtime contention. The
candidate is therefore not credited with a new camera tier. It is retained for
the deterministic CPU-copy and allocation reduction, with a slightly lower
median and maximum in this sample.

## Validation

- A prepared Requests body keeps the memoryview type and reports the full
  `nbytes` as Content-Length.
- A focused payload test covers a non-contiguous source, byte-for-byte equality,
  width, height, and result parsing.
- 55 remote-transport and grouped-inference tests passed.
- Two candidate live stress runs produced zero HTTP/inference failures.
