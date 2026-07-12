# Jetson cross-camera microbatch experiment — 2026-07-12

## Decision

Reject model-server-side cross-camera microbatching for the current edge
admission architecture. A fixed batch-2 YOLO26 Small INT8 engine improves
isolated aggregate throughput, but waiting for a second HTTP request after both
requests have consumed edge admission slots increases queueing and loses the
existing fourteen-camera zero-drop tier.

Production remains on the calibrated batch-1 YOLO26 Small INT8 640px engine,
three edge admission slots, a 75 ms bounded admission wait, and evenly staggered
camera phases. No Nano model was used.

## Fixed batch-2 engine

The batch-2 candidate used the same YOLO26 Small source and calibration corpus
as the deployed batch-1 engine. TensorRT consumed 347 valid calibration frames;
24 static JPEGs were skipped because Ultralytics attempted an in-place EXIF
repair while the corpus was mounted read-only. The build completed in 451.0
seconds and produced a 10.8 MiB engine.

Two-frame isolated benchmarks used four representative office frames, ten
warmups, and 200–300 timed pairs at 640px and confidence 0.10.

| Precision | Execution | Throughput | Pair median | Pair p95 |
| --- | --- | ---: | ---: | ---: |
| FP16 | two batch-1 calls | 66.399 FPS | 30.185 ms | 30.486 ms |
| FP16 | one batch-2 call | 76.619 FPS | 26.094 ms | 26.469 ms |
| INT8 | two batch-1 calls | 81.507 FPS | 24.626 ms | 24.916 ms |
| INT8 | one batch-2 call | 99.784 FPS | 20.132 ms | 20.615 ms |

Batch-2 improved same-precision FP16 throughput by 15.4% and INT8 throughput
by 22.4%. The INT8 batch-1 and batch-2 engines returned exactly matching class
IDs, confidences, and boxes on all four control frames.

## Exact edge-load result

The candidate model server paired only compatible singleton COCO requests and
fell back to batch-1 after a bounded wait. The load generator exercised the
real `model_manager.predict_record_batches()` transport with the calibrated
640px primary, 11.1% PPE duty, phone/PPE overlap avoidance, and 4 FPS per
camera.

| Camera phase / batch wait / edge slots | Completed | Drops | Specialist calls | Pairing | Median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current batch-1 baseline, staggered / none / 3 | 1680 / 1680 | 0 | 182 | n/a | 17.484 ms | 70.876 ms |
| Staggered / 6 ms / 3 | 1675 / 1680 | 5 | 179 | 0 / 1496 | 25.082 ms | 97.835 ms |
| Staggered / 20 ms / 3 | 1676 / 1680 | 4 | 181 | 1478 / 1495 | 42.488 ms | 102.126 ms |
| Paired / 6 ms / 3 | 1672 / 1680 | 8 | 180 | 1308 / 1492 | 27.780 ms | 100.648 ms |
| Paired / 6 ms / 4 | 1675 / 1680 | 5 | 181 | 1320 / 1494 | 27.962 ms | 110.320 ms |

Six milliseconds cannot pair evenly staggered fourteen-camera arrivals because
they are about 17.9 ms apart. A 20 ms wait pairs 98.9% of eligible requests but
costs more queue time than the batch saves. Pairing camera phases removes most
of that wait, but each frame has already acquired an edge admission slot, so
the paired bursts still shed work with either three or four slots.

The isolated engine result is real, but it does not establish a production
capacity improvement. A future batching attempt must pair frames on the edge
*before* remote admission and send both frames in one request that consumes one
bounded slot. That is a different transport design and needs its own failure,
timeout, rolling-upgrade, and alert-freshness validation rather than promotion
of this server-side queue.
