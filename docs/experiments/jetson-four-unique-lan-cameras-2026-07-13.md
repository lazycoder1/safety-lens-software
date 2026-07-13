# Jetson four-unique-LAN-camera validation — 2026-07-13

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, TensorRT 8.5.2.2.

## Decision

The live Rakshak Lens application now supports every enabled camera found on
the Techser office camera LAN: four distinct CP Plus camera scenes through the
existing NVR. A 600-second full-application soak sampled health 273 times. All
273 samples were `ok`; all four workers remained frame-fresh on
`gstreamer_nvdec`; and there were zero camera reconnects, inference failures,
overload drops, inference-route failures, alert persistence failures, alert
delivery failures, or queue buildup.

A focused follow-up moved FRONT/cam2 from the NVR's 352×288 substream to its
main stream while retaining direct NVDEC/VIC output at 640×360. All four live
MJPEG endpoints now publish 640×360. The post-restart acceptance run kept all
four workers fresh on hardware decode with zero connection failures, inference
failures, or overload drops.

This raises the unique-camera evidence from two to four. It does **not** turn
the earlier 25-connection resource envelope into a 25-distinct-camera
certification. Four is the number of unique enabled cameras available on this
LAN. A larger distinct-feed claim still needs a larger site or temporary camera
pool and a multi-hour full-stack soak.

The configured four-FPS inference value is a ceiling, not a constant live
rate. Production motion-adaptive inference skips visually unchanged frames,
forces at least a one-FPS heartbeat, and bypasses the skip while an alert needs
confirmation. During this real CCTV soak, completed inference averaged
1.228–2.943 FPS depending on scene motion. No work was dropped through
admission pressure.

## LAN inventory

The Jetson physical interface is attached to one private `/24` LAN. Credential-
free RTSP discovery found five RTSP devices:

- one NVR; and
- four distinct CP Plus IP cameras registered to that NVR.

The NVR's authenticated remote-device inventory reported exactly four enabled
camera slots and four disabled/empty slots:

| NVR channel | Camera label | Device family | Enabled |
| ---: | --- | --- | --- |
| 1 | GGS | CP-UNC-DA20L3S-V2 | yes |
| 2 | FRONT | CP-UNC-TA20L3S-V2 | yes |
| 3 | STORE | CP-UNC-TA20L3S-V2 | yes |
| 4 | BIKE PARKING | CP-UNC-TA20L3S-V2 | yes |

The direct IP cameras require credential sets different from the credentials
already stored for the NVR. No camera or NVR credentials were printed,
exported, or copied. The two missing scenes were therefore added through NVR
channels 3 and 4 using the existing credential boundary and authenticated
camera-create API. A mode-0600 private runtime-config backup was created before
the import.

## Live camera configuration

| Camera ID | Scene | Source shape published | AI target | Selected rules |
| --- | --- | ---: | ---: | --- |
| cam1 | GGS / NVR channel 1 | 640×360 | 4 FPS ceiling | person, phone, animal |
| cam2 | FRONT / NVR channel 2 | 640×360 main stream | 4 FPS ceiling | phone, animal, rider helmet |
| cam3 | STORE / NVR channel 3 | 640×360 | 4 FPS ceiling | person, phone, animal |
| cam4 | BIKE PARKING / NVR channel 4 | 640×360 | 4 FPS ceiling | phone, animal, rider helmet |

All four current MJPEG endpoints rendered at 640×360. A fresh frame from each
was reduced to the same 32×18 grayscale representation; all four checksums
were unique. The earlier perceptual-hash validation also found every pair
distinct, with Hamming distances from 23 to 38. Together these reject an
accidental duplicate-channel configuration before and after the source change.

## Uniform 640 rollout follow-up

The earlier FRONT animal alert exposed how little object detail remained in the
352×288 substream. The authenticated camera-update API changed only cam2's
NVR path to channel 2 main and its preferred profile to `main`; its three safety
rules and credentials were preserved. A private mode-0600 runtime-config backup
was created before the update, and the normal camera restart path applied the
change.

The runtime still caps capture output at 640 pixels wide, so the application is
not moving the NVR main-stream resolution through the rest of the pipeline. It
decodes and scales once in the NVDEC/VIC path, publishes 640×360, and feeds the
models at their native 640 input. This removes the preventable loss of source
detail without increasing model tensor size.

The focused acceptance sampled `/api/health` 30 times at ten-second intervals.
All 30 samples were `ok`; all four cameras were frame-fresh, worker-alive,
`gstreamer_nvdec`, and hardware-acceleration-active. Total camera failures,
inference failures, and overload drops remained zero. FRONT completed 795
additional inference passes between the first and last samples, or about 2.74
effective FPS under the same four-FPS motion-adaptive ceiling. Primary, PPE,
specialist, and RT-DETR admission/route failure deltas were all zero. Persistence
accepted four alerts during the wider 353-second comparison window with zero
persistence or delivery failures and zero queue depth. Their semantic accuracy
was not judged in this focused infrastructure soak.

This is a source-quality fix, not proof that animal false positives are
eliminated. The same pose did not recur during the acceptance window. A
site-labelled replay remains the correct gate for any animal confidence or
edge-truncation rule change.

## Ten-minute full-application soak

Acceptance watched the real `/api/health` payload every two seconds while the
normal camera workers, model server, stream publication, rules, persistence,
and alert workers remained active. Raw result and `tegrastats` evidence are
retained on the Jetson under:

`/opt/rakshak-lens/model-server-models/experiments/lan-four-unique-2026-07-13/`

| Camera | Completed inference | Effective live FPS | Max frame age | Decode backends | Failures / overloads / reconnects |
| --- | ---: | ---: | ---: | --- | ---: |
| cam1 | 791 | 1.318 | 1.0 s | NVDEC only | 0 / 0 / 0 |
| cam2 | 1,757 | 2.928 | 1.1 s | NVDEC only | 0 / 0 / 0 |
| cam3 | 737 | 1.228 | 1.1 s | NVDEC only | 0 / 0 / 0 |
| cam4 | 1,766 | 2.943 | 1.1 s | NVDEC only | 0 / 0 / 0 |

Every sample was fresh and worker-alive. All primary, PPE, specialist, and
RT-DETR transport admission-overload and route-failure deltas were zero.
Persistence, delivery, and retry queues never exceeded depth zero.

The original automated gate marked the run `pass=false` only because it
required every scene to complete at least 3.8 inference FPS. That requirement
is intentionally incompatible with the production unchanged-scene skip. The
physical/full-stack health gate passed; the constant-rate assertion is
rejected and replaced by the observed adaptive rates above.

## Alerts

The stored alert count rose from 232 to 235 during the soak. The three new
events were real `Person Detected` P4 alerts from cam1 and cam3. Alert
persistence stayed healthy with no failures.

Only the in-app alert output is configured and ready. Browser sound is
simulated and limited to P1. Telegram, email, webhook, Pushover/iPhone, speaker,
and relay outputs are disabled or need setup. Consequently, a successful
detection/alert does not currently create a mobile notification. That is an
output-configuration fact, not an inference-capacity failure.

## Jetson telemetry

The soak retained 592 one-second `tegrastats` samples.

| Metric | Mean | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Jetson RAM used | 6,398.82 MB | 6,411 MB | 6,414 MB |
| Swap used | 2,066.44 MB | 2,067 MB | 2,067 MB |
| GPU utilization | 5.96% | 46% | 94% burst |
| Input power | 8.10 W | 8.40 W | 8.67 W |
| CPU temperature | 66.46°C | 66.75°C | 66.97°C |
| GPU temperature | 65.02°C | 65.31°C | 65.66°C |

The edge container averaged 19.27% CPU and 225.72 MiB cgroup memory. The model
server averaged 7.72% CPU and 1,085.54 MiB cgroup memory. RAM and swap did not
grow during the run, but the system began with only about 0.9 GiB available
memory and already-occupied swap. Four cameras have ample compute headroom;
whole-device unified-memory headroom remains the nearer expansion risk.

## Correct capacity statement

- **Live distinct cameras proven:** four, covering every enabled camera on the
  current Techser camera LAN.
- **Per-camera inference behavior:** configured ceiling four FPS; measured
  adaptive completion 1.228–2.943 FPS on the observed scenes, with a one-FPS
  unchanged-scene heartbeat.
- **Synthetic/connection resource envelope:** 25 simultaneous connections
  under the earlier strict mixed-model harness.
- **Not yet certified:** any count above four distinct cameras, and any
  20–25-camera multi-hour production claim.
