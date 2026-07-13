# Jetson physical camera capacity — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

The repeatably validated physical tier is **25 camera connections** with:

- 640-pixel-wide NVDEC output;
- a six-FPS RTSP capture cap that never reduces the four-FPS AI cadence;
- YOLO26 Small INT8 primary inference at four effective decisions per second;
- worst-case one-FPS PPE confirmation on every camera; and
- one device-wide RT-DETRv4-S phone-recall FPS.

Twenty-six cameras completed one sustained combined run, but a second
independent attempt could not reopen every duplicated NVR session. It remains
provisional rather than supported. Twenty-seven is rejected because even at
four capture FPS and direct 640-pixel output it produced 84 inference overloads
and a 340.290 ms PPE tail.

The earlier 27-camera result was an inference-only camera-equivalent tier. It
did not include RTSP decoding, NVDEC/VIC transfers, frame copies, or NVR session
startup. This experiment supersedes that number for physical deployment
planning.

Only four distinct office NVR channels were reachable. Higher connection counts
therefore reused those feeds round-robin to measure Jetson capture and shared
resource pressure. This is valid device-load evidence, but repeated teardown
also exposed a source-side NVR session-table limit that would not occur with 25
independent cameras or channels.

## Engineering bottlenecks found

1. **Inference-only capacity was being treated as camera capacity.** At 27
   camera-equivalents inference passed in isolation, while combined physical
   capture caused hundreds of overloads.
2. **All camera workers opened RTSP concurrently.** Concurrent 29-stream probes
   were unreliable, while 50–250 ms staggered probes opened all 29 every time.
3. **The capture path produced 960-wide frames for 640-input engines.** Direct
   640 output removed a redundant VIC/host resize and materially reduced CPU
   and memory without changing model input resolution.
4. **Eight capture FPS moved twice as many frames as the four-FPS detector
   needed.** A six-FPS cap retains live-view headroom while removing
   non-inference transfers. The cap is not allowed below the configured AI FPS.
5. **The old NVDEC benchmark used one duplicated source and omitted memory.** It
   now supports credential-safe channel distribution, bounded open staggering,
   and peak RSS reporting.

## Reachable physical sources

Channels 1–4 returned frames. Channels 5–16 did not. Full credential-bearing
URLs were never emitted by the inventory or benchmark. The lower NVR stream was
352×288; the production-aligned main stream produced 960×540 before the
selected 640×360 capture resize.

## Decode-only boundary

All passing cases used the four reachable main streams round-robin, an
eight-FPS target, 960-pixel maximum dimension, and decoder drop interval three.

| Connections | Opened | Minimum FPS | Aggregate FPS | Process CPU | Peak RSS | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 4 | 8.040 | 32.461 | 17.01% | 181 MB | pass |
| 8 | 8 | 8.104 | 65.169 | 34.31% | 255 MB | pass |
| 12 | 12 | 8.013 | 97.030 | 52.34% | 315 MB | pass |
| 16 | 16 | 8.087 | 129.940 | 72.50% | 449 MB | pass |
| 20 | 20 | 7.790 | 157.203 | 80.81% | 573 MB | pass |
| 24 | 24 | 7.791 | 189.343 | 86.74% | 599 MB | pass |
| 28, run 1 | 28 | 7.722 | 219.643 | 90.89% | 680 MB | pass |
| 28, run 2 | 28 | 7.733 | 219.316 | 90.30% | 676 MB | pass |
| 29, concurrent open | 27 / 29 | — | — | — | — | reject startup |

At 29 requested streams, every tested stagger from 50 to 250 ms opened all 29.
The selected production startup delay is 100 ms, which opened them in 3.997
seconds and avoids the simultaneous NVR/NVDEC allocation storm.

## Combined physical capture and inference

Every inference case below included all-camera one-FPS PPE confirmation and one
aggregate RT-DETR FPS. The freshness limit is 250 ms.

| Cameras | Capture profile | Duration | Decisions | Drops / failures | Primary max | PPE max | RT max | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 960 / 8 FPS | 30 s | 2,400 / 2,400 | 0 / 0 | 60.944 ms | 96.283 ms | 71.683 ms | pass |
| 24 | 960 / 8 FPS | 30 s | 2,880 / 2,880 | 0 / 0 | 132.988 ms | 199.690 ms | 115.378 ms | pass |
| 25 | 960 / 8 FPS | 30 s | 3,000 / 3,000 | 0 / 0 | 169.652 ms | 258.456 ms | 88.171 ms | reject tail |
| 25 | 640 / 8 FPS | 30 s | 3,000 / 3,000 | 0 / 0 | 168.576 ms | 242.286 ms | 90.873 ms | provisional |
| 25 | 640 / 8 FPS | 60 s | 6,000 / 6,000 | 0 / 0 | 143.432 ms | 253.534 ms | 88.265 ms | reject tail |
| 26 | 640 / 8 FPS | 30 s | 3,120 / 3,120 | 0 / 0 | 180.457 ms | 254.838 ms | 96.431 ms | reject tail |
| 27 | 640 / 4 FPS | 30 s | 3,156 / 3,240 | 84 / 0 | 267.093 ms | 340.290 ms | 137.434 ms | reject |
| 25 | 640 / 6 FPS | 60 s | 6,000 / 6,000 | 0 / 0 | 143.916 ms | 231.517 ms | 89.659 ms | pass |
| 25, sustained run 1 | 640 / 6 FPS | 60 s | 6,000 / 6,000 | 0 / 0 | 141.232 ms | 222.257 ms | 87.926 ms | pass |
| 25, sustained run 2 | 640 / 6 FPS | 60 s | 6,000 / 6,000 | 0 / 0 | 152.734 ms | 231.331 ms | 88.718 ms | pass |
| 26 | 640 / 6 FPS | 60 s | 6,240 / 6,240 | 0 / 0 | 166.410 ms | 230.636 ms | 92.555 ms | provisional |

The two sustained 25-camera repeats shared one stable set of capture sessions.
All 25 connections remained open, delivered a minimum 5.942 FPS, and produced
149.339 aggregate FPS. Capture used 72.33% of one CPU core and 448.973 MB peak
RSS. Both inference repeats completed at exactly four effective FPS per camera.

## Accuracy preservation

The 960 and 640 capture shapes were replayed through the actual Jetson worker
and alert evaluators:

- RT-DETR phone outcomes remained 5/6 actionable positives and 0/4 negatives;
- every per-image actionable phone result was identical between widths;
- all 18 PPE corpus frames passed both worker paths at both widths; and
- all 18 PPE violation-rule outcomes were identical between widths.

The capture change therefore removes a redundant pre-engine resolution rather
than lowering the 640-pixel model input.

## Runtime contract

Generic deployments remain unchanged by default. The measured Jetson profile
uses:

```dotenv
SAFETYLENS_CAMERA_STARTUP_STAGGER_SECONDS=0.10
SAFETYLENS_RTSP_MAX_DIMENSION=640
SAFETYLENS_RTSP_CAPTURE_FPS_CAP=6
```

Raw evidence is stored on the Jetson under:

- `/opt/rakshak-lens/model-server-models/experiments/nvdec-physical-sweep/`
- `/opt/rakshak-lens/model-server-models/experiments/physical-combined-cases/`
- `/opt/rakshak-lens/model-server-models/experiments/physical-25-dual-inference/`
- `/opt/rakshak-lens/model-server-models/experiments/capture-resize-parity/`

## Live rollout

Commit `36b2cfc` was pushed directly to `master`, and its hash-matched edge
image was promoted with the three measured Jetson settings. The model server
remained warm and the prior edge container remains stopped as rollback.

After the reconnect cycle:

- overall edge health was `ok` with no reasons;
- cam1 and cam2 were both running, frame-fresh, and using `gstreamer_nvdec`;
- the effective capture rate resolved to six FPS for both configured cameras;
- cam1's published main-stream frame was 640×360, confirming direct 640-wide
  capture, while cam2 retained its native 352×288 substream;
- the cameras had 287 inference successes at the observation point with zero
  failures or overload drops;
- primary, PPE, and RT-DETR transports had zero admission or route failures;
- alert persistence and delivery workers were alive with zero failures; and
- the one-minute model watchdog timer was active.
