# Jetson multi-camera pipeline, scheduler, and VLM evaluation — 2026-07-15

Target: NVIDIA Jetson Orin NX Developer Kit, JetPack 5.1.3 / L4T R35.5,
MAXN, 8 GB unified memory.

## Decision

Keep YOLO26 Small TensorRT as the real-time detector. The next material
throughput improvement is pipeline scheduling, not another detector or an
on-device VLM.

- Retain direct 640-wide NVDEC/VIC analytics output. Existing parity evidence
  shows no phone/PPE rule change versus the previous 960-wide capture path.
- Retain bounded admission and the latest-frame GStreamer appsink. Do not add
  another FIFO camera queue.
- Promote adaptive batch-4 as the preferred inference shape when arrivals can
  be coordinated. In the controlled matrix it reduced p99 scheduled-frame age
  about 19–27% versus singleton execution while preserving throughput and
  fairness.
- Build the next scheduler as one central, fair, timestamp-aware owner of one
  replaceable frame slot per camera. The current implementation is a shared
  rendezvous entered by camera threads; it is not a central scheduler.
- Keep scene-adaptive rates. A repeating 1/2/4 FPS workload completed every
  request with demand-normalized Jain fairness 1.0. Add a true
  quiet/uncertain/active controller only with immediate escalation for new
  entry, low tracker confidence, zone entry, or possible violation.
- Do not promote the tested ByteTrack configuration as a detector-skip filter.
- Do not promote person-crop phone inference. It added work without recovering
  a miss. The validated PPE optimization is full-frame PPE-only substitution
  combined with cached primary context; it is not person-crop PPE inference.
- Keep all VLM work outside the Jetson real-time failure domain. A separate
  thread or container on the same 8 GB device is insufficient.

The supported external capacity statement remains conservative: four distinct
live RTSP cameras have a ten-minute full-stack proof. Higher numbers below are
short repeated-source or inference-only resource envelopes, not certification
of that many distinct cameras, networks, or scenes.

## Safety and method

The existing production master was pushed and verified before this experiment.
The Qwen trial was already removed. The isolated scheduler tests stopped only
the edge container; the warm production model server, database, and frontend
remained in place. No camera configuration, model, image, token, or container
definition was changed.

The new load report separates:

- scheduled work from work completed inside the measurement window;
- post-window drain from capacity;
- stale-before-submit drops from admission overloads and failures;
- raw FPS fairness from demand-normalized fairness;
- inter-completion gaps from an edge-inclusive maximum service gap; and
- observed inference latency from external substitutions whose latency is not
  measurable.

Raw results and resource logs are stored on the Jetson under:

`/opt/rakshak-lens/benchmarks/20260715-pipeline-goal/`

They contain no RTSP URLs, camera credentials, model-server token, or frames.

## Final restore gate

The exact original edge image and runtime config were restored:

- edge image: `rakshak-lens-edge:candidate-physical-capacity-36b2cfc`, running,
  restart count zero;
- model-server image: `rakshak-lens-model-server:candidate-ppe-only-route`,
  running, restart count one (the retained Qwen OOM evidence); and
- runtime config SHA-256:
  `f6a8f425ad790ebbaef5e7cad9e9887b00a287cf0450f37cd090fc1246282fc4`.

The four-camera restore gate did **not** pass. After separate 90- and
180-second edge-off cooldowns, cam1 briefly opened through FFmpeg and then its
RTSP source went stale again. In the final 30-second measurement, cam1 was
stale in 15/15 health samples and completed no inference. Cam2 and cam3 stayed
fresh on NVDEC, cam4 stayed fresh on FFmpeg, and the three active channels
completed 258 inferences with zero inference failures, overload drops, or new
connection failures. Mean RAM was 5,765 MB, mean input power was 8.41 W, and
the collector correctly returned a failed gate.

The unchanged deployment was left running with bounded reconnect retries. The
remaining blocker is the cam1/NVR source path, not an unreverted model, config,
container, or VLM artifact. It requires camera/NVR-side investigation before a
four-camera healthy state can be claimed.

## Live four-camera baseline

The retained baseline sampled 90 health snapshots over 407 seconds and 179
`tegrastats` samples. Every one of 360 camera-health observations was fresh.
There were no new reconnect failures, inference failures, overload drops, or
stale health observations. Cam4 was on the existing FFmpeg fallback; the other
three used NVDEC. The runtime does not expose an appsink/decode dropped-frame
counter, so capture drops cannot be claimed from this health window.

| Camera | Effective inference FPS |
| --- | ---: |
| cam1 | 0.902 |
| cam2 | 3.083 |
| cam3 | 0.983 |
| cam4 | 2.897 |
| Total | 7.865 |

The fixed ceiling would be 16 FPS. The current binary scene-adaptive behavior
therefore avoided about 51% of primary calls in this scene mix. Raw Jain
fairness was 0.786, but raw fairness is the wrong SLA when camera demand is
different; the controlled mixed-rate test below uses demand-normalized
fairness.

The baseline scheduler used batch-2 for 484 of 3,192 eligible frames (15.2%),
batch-4 for none, and timed out 2,708 frames (84.8%) to singleton execution.
This is the clearest evidence that the deployed camera-thread rendezvous leaves
batch capacity unused on real, asynchronous scenes.

Baseline resources:

| Metric | Mean | p95 | Maximum |
| --- | ---: | ---: | ---: |
| RAM used | 5,943 MB | 5,987 MB | 5,991 MB |
| Swap used | 1,440 MB | — | — |
| GPU | 5.8% | 45.1% | 68% |
| VIC | 11.6% | 23% | 25% |
| Input power | 8.49 W | 8.75 W | 9.31 W |

CPU and GPU averaged 67.45 C and 65.84 C in the retained baseline. Maximum
temperatures stayed below 71 C in these runs; no explicit thermal-throttle
telemetry was collected.

## Shared scheduler A/B

Each case used four distinct saved frames, 640 input, four detector decisions
per second per camera-equivalent, no specialists, a 15-second measurement
window, two bounded in-flight jobs, and grouped camera phases. All p99s contain
at least 240 samples. “Scheduled-slot age” starts at the harness's intended
dispatch time and ends at inference completion; it is not sensor/capture
end-to-end age.

| Equivalent cameras | Scheduler | Completed in window | Aggregate FPS | p99 scheduled-slot age | Fail / overload | Demand fairness |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | singleton | 240 / 240 | 16.0 | 57.8 ms | 0 / 0 | 1.000 |
| 4 | fixed batch-2 | 240 / 240 | 16.0 | 49.7 ms | 0 / 0 | 1.000 |
| 4 | adaptive batch-4 | 240 / 240 | 16.0 | 44.7 ms | 0 / 0 | 1.000 |
| 12 | singleton | 720 / 720 | 48.0 | 56.5 ms | 0 / 0 | 1.000 |
| 12 | fixed batch-2 | 720 / 720 | 48.0 | 50.0 ms | 0 / 0 | 1.000 |
| 12 | adaptive batch-4 | 720 / 720 | 48.0 | 45.5 ms | 0 / 0 | 1.000 |
| 20 | singleton | 1,199 / 1,200 | 79.9 | 58.1 ms | 0 / 0 | 0.99999 |
| 20 | fixed batch-2 | 1,200 / 1,200 | 80.0 | 49.9 ms | 0 / 0 | 1.000 |
| 20 | adaptive batch-4 | 1,200 / 1,200 | 80.0 | 42.6 ms | 0 / 0 | 1.000 |

At 20 equivalents, adaptive batch-4 reduced p99 scheduled-slot age 26.6% versus
singleton and 14.6% versus batch-2. Mean input power was 11.57 W for adaptive
batch-4 versus 11.97 W for singleton. Adaptive batch-4 used 5,338 MB mean RAM,
reached 68.9 C
CPU / 68.3 C GPU maximum, and had no restart or OOM.

This controlled result is a best-case coordinated-arrival test. Real production
traffic is not this aligned, which is why a central scheduler with per-camera
latest slots is preferable to asking camera threads to rendezvous.

## Primary-only inference boundary

The same adaptive batch-4 workload was extended until the first clear knee.

| Equivalent cameras | Target FPS | In-window completion | Delivered FPS | Overloads | p99 scheduled-slot age | Demand fairness |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 96 | 100.0% | 96.0 | 0 | 45.3 ms | 1.0000 |
| 28 | 112 | 99.76% | 111.7 | 0 | 53.1 ms | 0.99997 |
| 32 | 128 | 99.79% | 127.7 | 0 | 49.9 ms | 0.99997 |
| 36 | 144 | 92.59% | 133.3 | 148 | 126.9 ms | 0.99855 |
| 40 | 160 | 83.50% | 133.6 | 384 | 127.0 ms | 0.99742 |
| 44 | 176 | 76.21% | 134.1 | 616 | 126.6 ms | 0.99826 |
| 48 | 192 | 69.72% | 133.9 | 860 | 127.5 ms | 0.99547 |

Throughput plateaus near 134 primary frames per second. Twenty-four is the last
strict 100%-inside-window tier. Thirty-two is the last zero-overload short tier:
1,916 of 1,920 requests completed inside the 15-second window and the remaining
four completed during bounded drain; 36 is rejected because real admission
overloads begin. At 32, input power averaged 14.26 W and reached 14.91 W, RAM
averaged 5,441 MB, GPU utilization averaged 61.3% and reached 98%, and CPU/GPU
maxima were 70.6/70.1 C.

This does **not** support 32 physical cameras. It omits decode, source/network
jitter, rule evaluation, PPE, phone recall, alert persistence, and long-duration
recovery. The stronger combined prior evidence is:

- four distinct LAN cameras: 600-second full-stack pass;
- eight hybrid sources: short pass with four real RTSP plus four files;
- ten hybrid sources: late source-recovery failure;
- 25 repeated NVR connections: two 60-second combined passes at 640-wide,
  six-FPS capture, four decisions/camera/second, one PPE FPS on every camera,
  and one device-wide RT-DETR FPS; and
- 27 camera-equivalents: inference-only specialist-inclusive result.

## Quiet / uncertain / active rates

Sixteen camera-equivalents used a repeating 1/2/4/1 FPS profile, representing
quiet, uncertain, active, and quiet states. The requested aggregate was 32 FPS.

| Scheduler | Completed | p99 scheduled-slot age | Batch-4 / batch-2 / timeout frames | Raw Jain | Demand Jain |
| --- | ---: | ---: | ---: | ---: | ---: |
| singleton | 480 / 480 | 57.0 ms | 0 / 0 / 0 | 0.727 | 1.000 |
| adaptive batch-4 | 480 / 480 | 43.6 ms | 240 / 120 / 120 | 0.727 | 1.000 |

Raw Jain correctly falls because requested FPS differs. Demand-normalized Jain
is 1.0 because every camera received its full requested service. Adaptive
batching reduced p99 scheduled-slot age 23.5% even though one quarter of
requests could not find a partner and timed out to singleton.

Production currently has a binary motion-adaptive cadence, not a three-state
controller. The live baseline demonstrates the savings, while this synthetic
case validates the proposed rates and fairness math. A production state machine
still needs transition telemetry and replay validation before promotion.

## Latest-frame and stale-work behavior

The NVDEC capture path already uses `appsink max-buffers=1 drop=true`, and the
FFmpeg fallback performs a bounded buffer drain. The model admission queue is
also bounded. Two deliberately overloaded tests therefore shed work at
admission before a per-camera schedule became stale:

| Workload | Stale threshold | Stale drops | Admission overloads | p99 schedule lateness | p99 scheduled-slot age |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 × 4 FPS, one in-flight slot | disabled | 0 | 899 | 1.2 ms | 106.6 ms |
| 40 × 4 FPS, one in-flight slot | 100 ms | 0 | 907 | 1.3 ms | 106.9 ms |
| 24 × 8 FPS, one in-flight slot | disabled | 0 | 1,377 | 1.3 ms | 107.8 ms |
| 24 × 8 FPS, one in-flight slot | 100 ms | 0 | 1,375 | 1.5 ms | 107.0 ms |

The stale threshold made no meaningful difference because no scheduled slot
became 100 ms overdue. This is a negative but useful result: do not claim a
latest-queue gain from this harness. The next scheduler should still attach a
capture timestamp, atomically replace each camera slot, and reject an old frame
again immediately before inference. That protects against future downstream
stalls without adding a queue today.

## Hardware decode and analytics resolution

The retained 640-wide NVDEC result is stronger than the interrupted rerun:

- 25 repeated live connections sustained a minimum 5.942 capture FPS and
  149.339 aggregate FPS with 72.33% of one CPU core and 448.973 MB peak RSS;
- 640-wide capture removed a redundant VIC/host resize;
- phone decisions were identical at 960 and 640 (5/6 actionable positives,
  0/4 negatives); and
- all 18 PPE frames and all 18 PPE rule outcomes were identical at both widths.

The new live restart briefly recovered all four channels on NVDEC, but a later
health window lost one NVR channel while inference remained clean. A direct
decode ladder could not reopen the NVR sessions and produced no valid decode
samples. It is excluded rather than converted into a compute claim. Source
session cleanup and reconnect behavior, not decode throughput, is the observed
operational limiter.

For cameras that expose a same-aspect-ratio substream, the next production A/B
should use the substream for analytics and the main stream only for recording,
snapshots, or escalation. Do not globally select a 352×288 substream until
small helmet/phone parity is established per camera.

## Tracking and specialist models

The retained ByteTrack replay used 24 frames from four NVR feeds:

| Path | Median | p95 | Useful tracked frames |
| --- | ---: | ---: | ---: |
| detector only | 16.87 ms | 20.51 ms | — |
| detector + ByteTrack | 17.68 ms | 39.02 ms | 1 / 24 |

The median overhead was only 0.81 ms, but p95 worsened by 18.51 ms and tracking
did not bridge detector misses. That configuration is rejected as a
detector-skip filter. Tracking can still be valuable for dwell, identity, and
zone continuity if the detector is forced immediately on new entry, low track
confidence, fast box change, zone entry, possible violation, and a maximum
keyframe interval.

The current phone person-crop rerun used ten labeled images. These are raw
image-level `cell_phone` presence scores at confidence 0.10; they do not apply
production person association, temporal confirmation, or rule logic:

| Path | Recall | Precision | Extra median inference |
| --- | ---: | ---: | ---: |
| full frame | 66.7% | 66.7% | — |
| person crop | 66.7% | 80.0% | 20.8 ms |
| full frame OR crop | 66.7% | 66.7% | 20.8 ms |

The crop recovered neither full-frame miss. This agrees with the larger
retained 20-pair result: 58.3% recall, 87.5% precision, zero of three
full-frame misses recovered, and one added negative. Person-crop phone is not
promoted. The 20.8 ms value is median extra crop work; one three-person image
needed about 51.9 ms of crop calls, illustrating the multi-person latency tail.

No retained person-crop PPE benchmark exists. The production PPE optimization
is full-frame PPE-only inference plus cached primary context. It has 18/18
detection and 18/18 rule parity in its limited corpus. A crop-PPE experiment
needs tracked, padded ROIs, boundary/full-frame fallback, and explicit
small-person ground truth before it can replace that path.

## Qwen3-VL-2B

The Qwen3-VL-2B-Instruct family is attractive for asynchronous scene
descriptions, OCR, or operator review, but the tested co-resident deployment
does not make sense on this 8 GB Jetson beside the real-time detector. This was
a Q4_K_M GGUF plus Q8 vision-projector llama.cpp deployment, not the official
reference or full-precision runtime.

The isolated Q4_K_M model plus Q8 vision projector used CPU llama.cpp with
three threads. One request took 39.28 seconds and peak RSS was 1.70 GiB. Global
memory pressure killed and restarted the production model server, causing 206
camera inference failures. The one output also marked every queried PPE
attribute true, so it provided no accuracy evidence.

The rollback restored exact config/container hashes, removed the temporary
runtime, models, and images. In the 89-second rollback gate, all cameras were
fresh and the model server was healthy; overall status remained degraded only
for the pre-existing cam4 FFmpeg fallback. Moving this VLM shape to another
thread or container on the same device does not remove the shared-memory
failure domain. Use a remote GPU/service or separate hardware, sample only
event frames/crops, enforce a queue age limit, and never gate an alert on the
VLM response.

## Observability result and remaining gap

The new reusable collector safely records camera freshness and counter deltas,
per-camera FPS, raw Jain fairness, RAM, swap, GPU, VIC, power, thermals, and
container CPU/memory. It uses actual time between successful health samples,
rejects cumulative-counter resets, treats an absent expected camera as a failed
gate, bounds evidence fields to non-secret metrics, and truncates raw output on
each run.

Three requested metrics cannot be truthfully reported from the current runtime:

- capture/sensor timestamp to inference completion; and
- first positive rule observation to durable alert persistence/delivery p95/p99;
  and
- capture-side dropped frames inside appsink/decoder buffers (admission overload
  and stale-work drops are measured separately).

Health `lastFrameAgeSeconds` measures publication freshness, not sensor PTS.
Across the collected live windows only one natural alert occurred—none in the
retained baseline—and that separate post-restart window had a stale camera.
This is far short of a latency distribution, and the health payload exposes no
rule-first-seen or alert-persisted timestamps. Reporting p95/p99 would be false
precision. The next instrumentation patch should propagate a monotonic capture
timestamp through inference, count capture replacement/drop events, and attach
first-positive, confirmation, persistence, and delivery timestamps to an
isolated replay run with at least 100 alerts.

## How production-oriented systems and reference architectures differ

- NVIDIA DeepStream's reference architecture puts NVDEC/VIC, source
  multiplexing, frame metadata, and inference in one GPU-aware graph.
  `nvstreammux` forms batches across sources, uses a bounded formation timeout,
  and supports round-robin/priority policies. The current documentation is
  architecture guidance only for this Orin NX: DeepStream 8 targets newer
  JetPack hardware, while the archived DeepStream 6.3 material is the closest
  JetPack 5-era implementation reference.
- NVIDIA Triton owns one model queue and supports bounded-delay dynamic batches,
  queue size, priorities, and per-request timeouts. This is closer to the
  scheduler needed here than camera threads waiting for partners.
- Frigate recommends a separate low-resolution detection stream, commonly
  around five FPS, and a high-resolution recording stream. Its detector workers
  pull from a common queue across cameras and its pipeline uses motion regions
  before inference.
- DeepStream `nvinfer` secondary inference can operate on primary object crops. That is
  the right topology for PPE/phone crops, but this project's phone accuracy
  evidence currently rejects promotion.
- DeepStream's AI-agent reference example samples frames slowly and batches
  VLM work outside the real-time detector path. That is not an audited customer
  deployment, but it matches the required VLM isolation boundary.

Primary references:

- NVIDIA DeepStream overview (current architecture; not this target runtime):
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html>
- NVIDIA DeepStream 6.3 `nvstreammux` (closest JetPack 5-era reference):
  <https://archive.docs.nvidia.com/metropolis/deepstream/6.3/dev-guide/text/DS_plugin_gst-nvstreammux2.html>
- NVIDIA DeepStream `nvinfer` secondary inference:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html>
- NVIDIA DeepStream AI-agent reference example:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_AI_Agent_Skill.html>
- NVIDIA Triton dynamic batcher:
  <https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2670/user-guide/docs/user_guide/batcher.html>
- Frigate camera streams and video pipeline:
  <https://docs.frigate.video/frigate/camera_setup/> and
  <https://docs.frigate.video/frigate/video_pipeline/>
- Frigate common detector queue:
  <https://docs.frigate.video/configuration/object_detectors/>

## Recent papers worth testing, not blindly adopting

- **Ultralytics YOLO26** (June 2026), arXiv:2606.03748. The deployed YOLO26s
  belongs to a model family that captures the relevant NMS-free/export
  direction. The deployed engine manifest does not prove whether this export
  uses the paper's one-to-one NMS-free or one-to-many NMS path, and the paper
  does not replace this live multi-camera capacity test.
- **YOLOv13** (June 2025), arXiv:2506.17733. HyperACE and FullPAD improve COCO
  efficiency, but the reported environment does not establish Orin NX RTSP,
  batching, frame-age, or alert SLA behavior.
- **YOLOv12** (February 2025), arXiv:2502.12524. Attention-centric gains are
  interesting, but TensorRT/JetPack integration and site-domain accuracy would
  need a complete promotion gate.
- **YOLOE** (March 2025), arXiv:2503.07465. Open-vocabulary prompt embeddings
  are a more plausible asynchronous unknown-object experiment than a VLM in
  the primary loop.
- **MVP motion-vector propagation** (September 2025), arXiv:2509.18388. It runs
  a detector on keyframes and propagates detections with compressed-domain
  motion vectors, supporting a future tracking/keyframe experiment with
  explicit fallback.
- **See Without Decoding** (submitted 29 January 2026), arXiv:2602.00153. It reports
  compressed-domain tracking speedups with a modest accuracy tradeoff; this is
  research evidence, not a production SLA for the current cameras.
- **YOLO26 aquaculture benchmark** (10 July 2026), arXiv:2607.09835. Its useful
  result is that model generation alone did not decide the deployment tradeoff,
  but its hardware was an A100 and Raspberry Pi 5—not Jetson, RTSP, or a
  multi-camera scheduler.

These papers primarily optimize detector accuracy, per-frame efficiency, or
compressed-domain tracking throughput. None establishes this deployment's
distinct-camera count, source recovery, per-camera fairness, frame age, alert
latency, power, or thermal behavior. Pipeline measurements remain the
promotion gate.

## Implementation delivered

- `scripts/benchmark_conditional_model_server_load.py` now reports p95/p99/max
  latency and scheduled-frame age, mixed camera rates, stale-admission drops,
  bounded drain behavior, per-camera service gaps, in-window capacity, and raw
  plus demand-normalized Jain fairness.
- `scripts/benchmark_pipeline_soak_jetson.py` is a credential-safe Jetson soak
  collector for public health counters, Docker resource samples, and
  `tegrastats` power/thermal metrics.
- Tests cover percentile/fairness math, mixed-rate expansion, edge-inclusive
  starvation gaps, JetPack 5 parsing, secret exclusion, elapsed-time rates,
  counter reset rejection, and absent-camera failure.
- Ignore rules now cover atomic config temporaries, model payloads, private QA
  evidence, review captures, and the actual extras ZIP naming pattern.

## Next implementation order

1. Add monotonic capture, inference, rule-first-seen, persistence, and delivery
   timestamps; make the live health collector expose bounded histograms.
2. Replace camera-thread rendezvous with a central scheduler that owns one
   timestamped latest slot per camera, uses deficit/round-robin fairness, and
   gives urgent re-detection priority.
3. Add a per-camera quiet/uncertain/active state machine behind a feature flag;
   preserve the current cadence as the default.
4. Run an isolated repeated-violation replay to measure at least 100 complete
   alert latencies without polluting production alerts.
5. Test tracked, padded PPE crops with periodic full-frame fallback. Keep phone
   full-frame until a crop model beats the retained recall gate.
6. Keep VLM enrichment remote and asynchronous.
7. Investigate the cam1/NVR source session and add per-camera restart/cooldown
   control so one failed channel can recover without restarting healthy feeds.
