# Jetson hybrid full-application ladder: 6, 8, and 10 sources

Date: 13 July 2026  
Device: Techser NVIDIA Jetson Orin NX 8GB  
Code baseline: `0e5e48d` on `master`

## Question

Can the production application operate with 6, 8, or 10 configured sources when only four distinct office RTSP cameras are available?

This experiment is an application/load test, not a distinct-live-camera certification. The original four RTSP cameras remained configured. Temporary file cameras replayed TMEIC clips through normal camera workers, execution plans, tracking/rules, MJPEG publication, health, inference transport, and alert persistence.

## Sources and workload

- Live sources: `cam1` through `cam4`, four distinct NVR channels, 640×360 capture, NVDEC expected.
- Temporary sources: two alternating H.264 TMEIC clips, 854×640 at 8 FPS.
- TMEIC clip SHA-256 values:
  - Main gate: `05e1b981163bc3b85d00d2c4939235afb04e7ddb688efa764190a1f2ac70e5e2`
  - PE stores: `4bb3b172d6cae1fb3828ebb5709399f06d9721ee0c276c165aacbed68fa96a48`
- AI ceiling: 4 effective decisions per camera per second.
- Model placement: YOLO26 Small primary with context-gated PPE and RT-DETR specialists according to each camera's execution plan.
- Temporary alerts: rule thresholds were raised so the test exercised rule plans without polluting production alert history.
- Sample interval: 10 seconds.
- Pass requirements: every live camera fresh, every live camera on NVDEC, every temporary camera fresh, expected worker count present, no new inference/transport/pipeline failures, no overloads, and no queue buildup.

## Results

| Total sources | Composition | Samples / duration | Aggregate effective FPS | Effective FPS range | New connection failures | Inference / overload / transport / pipeline failures | Final queue | Result |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 6 | 4 RTSP + 2 files | 12 / 110.256 s | 16.407 | 1.959–3.220 | 0 | 0 / 0 / 0 / 0 | 0 | Pass |
| 8 | 4 RTSP + 4 files | 12 / 110.269 s | 20.513 | 1.197–3.111 | 0 | 0 / 0 / 0 / 0 | 0 | Pass |
| 10 | 4 RTSP + 6 files | 30 / 291.099 s | 25.203 | 1.123–3.164 | 14 | 0 / 0 / 0 / 0 | 0 | Fail |

The 10-source tier passed its first 24 samples. Samples 25–30 failed the full-stack health gate. All four RTSP feeds became stale, NVDEC activity fell away, and temporary file freshness also dipped before recovering. Maximum live frame age reached 4.8 seconds. The failure was not GPU admission, inference transport, alert persistence, or queue saturation.

One alert was persisted during the 10-source window: `Person Detected` on live `cam3` at 0.76 confidence. It was not a rider-helmet alert.

## Jetson telemetry

Telemetry was sampled once per second with `tegrastats`.

| Tier | RAM mean / max | Mean CPU per core / p95 | GPU mean / p95 / max | GPU temperature mean / max | Input power mean / max |
|---:|---:|---:|---:|---:|---:|
| 6 | 6186 / 6215 MB | 13.04% / 15.00% | 12.08% / 49% / 73% | 66.20 / 66.78°C | 9.06 / 9.52 W |
| 8 | 6306 / 6337 MB | 18.15% / 21.33% | 17.04% / 64% / 80% | 66.90 / 67.56°C | 9.65 / 10.52 W |
| 10 | 6427 / 6492 MB | 22.25% / 27.17% | 18.59% / 66% / 95% | 68.04 / 69.00°C | 10.09 / 11.31 W |

The 10-source failure occurred with substantial compute headroom. The leading observed bottleneck is source capture/recovery, not Small-model GPU throughput. CPU/application scheduling interaction and NVR behavior remain plausible triggers; this run does not isolate which failed first.

## UI finding

The frontend is served on port 3030 and the API on port 8000. Browser Use reached the login page and observed `Server ready`, confirming the prior `Connecting to server...` port-mapping incident is fixed.

The authenticated camera grid could not be exercised because the factory/default admin password no longer matched the deployed account. Source review shows:

- 2×2 grid: four visible cameras per page.
- 3×3 grid: nine visible cameras per page.
- Camera 10 appears on page 2.
- During the six-source tier, exactly six camera tiles were expected because only six cameras were configured at that point.

The grid should show an explicit `showing X of Y cameras` label so pagination is not mistaken for capacity.

## Rider-helmet false positives

The operator reports that every live `Missing rider helmet` alert is a false positive. Recent persisted examples came from `cam2` and remained stored with `falsePositive=false`.

The current rule treats absence of a helmet specialist detection as proof of a missing helmet after person/vehicle association. It then reports confidence from the person or vehicle detection, not confidence in the missing-helmet conclusion. A confident rider detection plus a missed helmet can therefore create a confident false alert.

Until site-labelled rider/helmet evidence passes, this rule should run in shadow mode or be disabled for customer-facing alerting. These alerts must not be counted as accuracy successes.

## Cleanup and recovery

- Temporary cameras `cam5` through `cam10` were deleted.
- Copied TMEIC assets were removed from the edge container.
- Exactly `cam1` through `cam4` remained configured.
- Inference and alert queues were empty after cleanup.
- Cameras 1, 3, and 4 recovered fresh on NVDEC promptly.
- Camera 2 remained in an RTSP reconnect loop after isolated restarts and a 30-second cooldown. Pausing only that worker for 90 seconds finally restored it fresh on NVDEC.
- All four original cameras finished fresh on NVDEC. Requiring manual extended cooldown still makes the 10-source tier an operational failure even though the model server stayed healthy.

## Conclusion

- Production claim: **4 unique live RTSP cameras proven**.
- Hybrid application evidence: **8 total sources passed for about 110 seconds**.
- Rejected: **10 total sources**, because it failed after roughly four minutes and destabilized a live RTSP source.
- Hardware-only evidence: **25 repeated connections** remains a short resource envelope, not a deployment count.
- Next gate: test 6 and then 8 distinct RTSP cameras for at least two hours each, including browser rendering, real rules, alert persistence/delivery, reconnect recovery, and thermal telemetry.

Raw Jetson evidence is retained at:

`/opt/rakshak-lens/model-server-models/experiments/full-app-tmeic-ladder-2026-07-13/`
