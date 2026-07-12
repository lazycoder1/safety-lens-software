# Jetson NVDEC runtime-preservation repair (2026-07-12)

Target: NVIDIA Orin NX Developer Kit, JetPack 5.1.3, GStreamer 1.16.3.

## Decision

Every Jetson edge-container recreation must preserve Docker
`HostConfig.Runtime=nvidia`. The checked-in Jetson Compose override now sets
`runtime: nvidia`, and `scripts/jetson_container_swap.py` clones the inspected
OCI runtime by default. `--runtime nvidia` can repair a container that was
previously recreated under plain `runc`.

## Root cause

The first version of the container-swap automation cloned mounts, NVIDIA media
devices, environment, ports, network, and restart policy, but omitted the OCI
runtime. A promotion therefore changed the edge from `nvidia` to the Docker
host's default `runc` runtime.

`NVIDIA_VISIBLE_DEVICES=all` and the eight explicit media-device mappings were
not sufficient by themselves. Without the NVIDIA runtime's CUDA device and
library injection, GStreamer's plugin scanner reported CUDA initialization
error 100 and blacklisted:

- `libgstnvvideo4linux2.so` (`nvv4l2decoder`)
- `libgstnvvidconv.so` (`nvvidconv`)

`nvdec_runtime_available()` then correctly returned false, so every RTSP open
fell back to CPU FFmpeg. The edge HTTP health endpoint still returned 200,
which proves process readiness alone is not an adequate post-promotion camera
gate.

## Repair validation

The corrected swap recreated the singleton-bypass candidate with
`Runtime=nvidia`. In the repaired container:

- all seven required GStreamer elements were discoverable;
- cam2 returned fresh on `gstreamer_nvdec` with zero inference failures;
- the model watchdog timer remained active;
- a credential-safe duplicate cam2 benchmark opened in 0.451 seconds;
- it delivered 122 frames in 15.118 seconds, or 8.07 FPS;
- `drop-frame-interval=3` was applied to `nvv4l2decoder`;
- the capture process used 0.377 CPU-seconds, or 2.49% of one CPU core.

The forward/rollback exercise then proved both the active singleton candidate
and the stopped prior adaptive image retained `Runtime=nvidia`. Cam1 continued
to flap between a brief fresh FFmpeg connection and its existing source outage;
that source-specific instability is independent of the plugin/runtime repair.

## Capacity interpretation

This repair restores the previously demonstrated physical-ingest tier rather
than claiming a new one. The earlier live gate sustained twelve simultaneous
RTSP/NVDEC pipelines at roughly 8 FPS using repeated office NVR feeds. Losing
the NVIDIA runtime silently invalidated that tier by moving decode back to CPU;
the corrected deployment path prevents that regression.

Post-promotion verification must check both:

1. the container HTTP health endpoint; and
2. expected fresh cameras reporting `captureBackend=gstreamer_nvdec` with no
   new inference failures.

The swap tool now accepts `--require-camera-fresh` and
`--require-camera-backend`; these conditions are part of its polling gate and
automatically restore the prior container if they remain false.
