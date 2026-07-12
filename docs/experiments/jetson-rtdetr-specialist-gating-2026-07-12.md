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

After deployment, a two-minute cam2 soak produced:

- 429 primary observations;
- 46 motorcycle-class frames and 45 person frames;
- 2 PPE specialist calls (0.47% duty, down from the 16.7% baseline sample);
- 1 allowed specialist frame with a helmet detection;
- zero inference overloads or failures on both production cameras.

The live scene was quieter than the baseline window, so capacity testing used the conservative 11.1% replay duty rather than 0.47%.

## Conditional-load capacity

The load test kept the two production RTSP cameras running and replayed four additional office-camera frames through the raw model-server path. Every virtual camera requested YOLO26 Small COCO at 4 FPS, a 960-pixel phone probe once per second, and YOLOE-26S PPE at 11.1% duty.

| Total camera-equivalents | Requests | Minimum FPS | Overloads | Failures | Median | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 960 | 4.00 | 0 | 0 | 21.63 ms | 41.92 ms |
| 7 | 1,200 | 4.00 | 0 | 0 | 21.31 ms | 44.21 ms |
| 8 | 1,440 | 4.00 | 0 | 0 | 21.51 ms | 43.32 ms |
| 9 | 1,680 | 4.00 | 0 | 0 | 21.45 ms | 44.78 ms |
| 10 | 1,915 / 1,920 | 3.95 | 5 | 0 | 22.14 ms | 52.71 ms |

The conservative zero-drop inference limit is therefore nine camera-equivalents at 4 detection FPS. Ten is a borderline operating point with 0.26% admission loss and is not promoted as clean capacity.

## Live decode check

Two additional NVR channels were temporarily added to the application, producing four simultaneous real RTSP workers. All four used `gstreamer_nvdec`, stayed frame-fresh, and recorded zero inference drops or failures over 60 seconds. Motion-adaptive achieved rates were 1.02 and 1.27 FPS on static scenes and 3.10 and 3.26 FPS on active scenes. A fifth named NVR channel did not provide a stream.

The site therefore validates four real simultaneous RTSP/decode pipelines. The nine-camera figure validates inference compute, not nine live decoders; a nine-channel NVR or RTSP simulator is still required for an end-to-end nine-stream soak. The temporary cameras were removed after the test.

Raw evidence remains on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/rtdetr/`

`/opt/rakshak-lens/model-server-models/experiments/gating/`
