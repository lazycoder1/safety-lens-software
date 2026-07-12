# Jetson RT-DETR and specialist-gating experiment (2026-07-12)

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

The deployed baseline was YOLO26 Small COCO at 640 pixels plus YOLOE-26S PPE at 640 pixels. Both production cameras remained online during the isolated experiments.

## RT-DETR result

The requested `RE-DETR` experiment was interpreted as Ultralytics RT-DETR. Ultralytics only provides pretrained Large and X weights for this family, so RT-DETR-L was the lighter available candidate. RF-DETR is a separate model family and should not be treated as the same artifact.

On 13 representative office/mobile-interaction frames at 640 pixels and confidence 0.35:

| Model | Parameters | Median | p95 | Peak PyTorch CUDA allocation |
| --- | ---: | ---: | ---: | ---: |
| YOLO26 Small PyTorch | 9.50 M | 32.90 ms | 265.45 ms | 232.6 MB |
| RT-DETR-L PyTorch | 32.15 M | 155.99 ms | 172.40 ms | 352.6 MB |

RT-DETR-L was 4.7 times slower by median. It found an additional phone in a laptop negative-control scene where the object was a landline/desk-phone shape, so the extra detections did not translate into a clean alert-quality win.

Two fixed-shape FP16 TensorRT export attempts were made in isolated, memory-capped containers:

- default ONNX simplification;
- simplification disabled with a 512 MB TensorRT workspace.

Both were killed by the Jetson unified-memory limit. No RT-DETR engine was promoted or deployed.

## Tracker result

ByteTrack was run with the deployed YOLO26 Small TensorRT engine over 24 frames from four reachable office NVR feeds.

| Path | Median | p95 |
| --- | ---: | ---: |
| YOLO26 Small prediction | 16.87 ms | 20.51 ms |
| YOLO26 Small + ByteTrack | 17.68 ms | 39.02 ms |

Median tracker overhead was 0.81 ms, but p95 increased by 18.51 ms and only one sparse frame produced a usable track ID. ByteTrack did not carry primary context through detector misses in this sample. It was not added to every production camera.

A one-second class-context hold was also replayed over 323 live cam2 observations. It increased PPE specialist duty from 16.7% to 22.0% without recovering an additional helmet event, so that variant was rejected.

## Primary-to-specialist gate result

The existing runtime already used COCO Small as the primary filter, but its rider gate accepted any motorcycle-class detection. The downstream rider alert evaluator was stricter: it required an evaluable person, an evaluable motorcycle, rider/vehicle geometric association, and PPE-zone scope. This mismatch invoked the specialist for observations that could never become alerts.

An 80-second live cam2 replay contained 323 observations and nine helmet-detection bursts:

| Gate | Specialist-eligible observations | Duty |
| --- | ---: | ---: |
| Existing previous-frame motorcycle gate | 54 | 16.7% |
| Previous-frame person + motorcycle evidence | 36 | 11.1% |

The stricter evidence preserved all nine bursts. It shifted the first helmet frame of one burst by one 250 ms inference interval, while the remaining four-frame burst stayed eligible.

The implemented gate goes further and exactly reuses the downstream size, confidence, person/vehicle geometry, and PPE-zone criteria. On the 24 extra office frames it suppressed the one old-gate specialist call; always-on specialist replay produced zero PPE detections on those frames.

Raw evidence remains on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/rtdetr/`

`/opt/rakshak-lens/model-server-models/experiments/gating/`
