# ANPR Real Footage Validation Plan

Date: 2026-06-20

## Current State

Local synthetic-gate evidence proves the plate-recognition runtime path, PaddleOCR PP-OCRv6 tiny recognizer wiring, plate-read persistence, API evidence, UI stream evidence, and detector-off schedule telemetry.

Local public-fixture evidence now also proves that the same split model-server path can read a public-domain Wikimedia Commons vehicle plate still encoded as a file-camera video. `anpr_public_plate_image_read` reads `MH01A5755` from the public-domain vintage MG image using the current plate detector plus PP-OCRv6 tiny, after a narrow Indian-format OCR normalization for common `O/0` and `I/1` confusions.

Local negative-control evidence now proves the same active `plate_recognition` pipeline emits zero detections and zero plate-read API rows on the no-plate office clip in `anpr_no_plate_false_positive_guard`, while schedule telemetry records an active-window plate-recognition invocation.

It does not prove production ANPR accuracy. The current sellable boundary remains:

- `anpr_synthetic_gate_plate_read`: ready for YAML/runtime wiring evidence.
- `anpr_public_plate_image_read`: ready for scoped public real-plate fixture evidence.
- `anpr_no_plate_false_positive_guard`: ready for scoped no-plate false-positive guard evidence.
- `anpr_detector_window_suppression`: ready for detector-off schedule evidence.
- Production ANPR on real gates: `needs_dedicated_scenario`.

## Web Search Findings

Public search did not identify a commercial-safe real gate video dataset that we can use immediately for production ANPR validation.

Checked sources:

- PaddleOCR / PP-OCRv6: https://github.com/PaddlePaddle/PaddleOCR and https://arxiv.org/abs/2606.13108. Use PP-OCRv6 tiny first, then PP-OCRv6 small if tiny misses too many reads.
- UFPR-VeSV: https://github.com/Lima001/UFPR-VeSV-Dataset. Real-world vehicle surveillance and ALPR annotations, but access requires a license agreement and is limited to academic non-commercial research.
- LPLC: https://github.com/lmlwojcik/lplc-dataset. Useful for license-plate legibility classes, but request-only and academic non-commercial.
- LPLCv2: https://github.com/lmlwojcik/LPLCv2-Dataset and https://arxiv.org/abs/2604.08741. Larger 2026 legibility dataset with 37,099 images and 41,487 annotated plates, but request-only and academic non-commercial.
- OpenALPR: https://github.com/openalpr/openalpr. Useful historical reference, but license/commercial constraints and age make it a poor replacement for the current PaddleOCR path.
- Wikimedia Commons public-domain vintage MG plate image: https://commons.wikimedia.org/wiki/File:%27Vintage_Mg_car%27_at_%27Mumbai_Vintage_car_rally-2010%27.jpg. Accepted as a scoped public fixture for runtime/OCR validation only; it is a still image encoded as a video, not gate footage, night/IR footage, motion blur, or a production dataset.

Do not use request-gated, academic-only, customer-private, or paid plate footage without approval.

## Required Footage

Minimum controlled gate pack:

- 30 minutes total footage from one fixed gate camera.
- 10 minutes daytime.
- 10 minutes night or low-light.
- 5 minutes vehicle motion blur or rolling approach.
- 5 minutes difficult cases: two-wheelers, trucks, dirty plates, glare, skew, partial occlusion, or unusual fonts.
- At least 100 readable plate passes with ground-truth plate text.
- At least 20 negative passes where no readable plate should be emitted.

Preferred production pack:

- 3 gate cameras.
- 2 hours per camera.
- Day, evening, night/IR, rain or wet-road glare if available.
- Front and rear plates where applicable.
- Passenger vehicles, commercial vehicles, two-wheelers, trucks, and delivery vehicles.
- Known allowlist, blocklist, visitor, and unknown vehicle examples.

Privacy and legal requirements:

- Written permission to use footage for model validation.
- Do not include driver identity, face, or unrelated sensitive areas unless masked.
- Store raw footage outside the repo if it contains real plates.
- Only commit derived metadata, redacted frame checks, manifests, and result summaries.

## Annotation Schema

For every plate pass:

```yaml
- clip_id: gate_cam_1_day_001
  camera_id: gate_cam_1
  start_time_seconds: 12.40
  end_time_seconds: 18.75
  expected_plate: KA05MN4523
  plate_region:
    frame_time_seconds: 15.20
    bbox_xyxy: [x1, y1, x2, y2]
  vehicle_type: car
  view: rear
  lighting: day
  motion_blur: low
  occlusion: none
  expected_event_type: plate_read
  list_status: unknown
  notes: readable on three or more frames
```

For negative passes:

```yaml
- clip_id: gate_cam_1_night_004
  camera_id: gate_cam_1
  start_time_seconds: 42.00
  end_time_seconds: 49.00
  expected_plate: null
  reason: unreadable_glare
  expected_event_type: no_plate_read
```

## YAML Scenario Plan

Use one detection at a time:

1. Create `qa/video_eval/focused/anpr_real_gate_active.yaml`.
2. Enable only `plate_recognition` for one gate camera.
3. Use PP-OCRv6 tiny first.
4. Run one positive clip with active detection window.
5. Run one negative or unreadable clip with no expected plate read.
6. Run one inactive detector window using the same readable clip and require zero plate candidates plus zero `plate_recognition` invocations.
7. If tiny fails on clear plates, rerun only after switching the YAML/model-server tier to PP-OCRv6 small and record the reason.

## Pass Gates

Minimum pilot gate:

- At least 95 percent read accuracy on clear, readable controlled passes.
- Zero plate reads in inactive detector-window runs.
- False-positive rate below 2 percent on negative/unreadable passes.
- Median OCR latency recorded separately from plate detection latency.
- One-gate Mac result recorded first.
- One-gate Jetson result recorded before claiming staging readiness.

Production gate:

- Accuracy reported separately for day, night/IR, motion blur, trucks, two-wheelers, and skewed plates.
- Stable under the deployed camera FPS and resolution.
- Allowlist/blocklist/visitor/unknown policy routing verified.
- External gate actions remain disabled until customer approval.
- 3-camera Jetson benchmark passes before any 3-camera ANPR claim.

## Open Ask

To move production ANPR beyond synthetic evidence, we need one of:

- Cleared customer gate footage matching the minimum controlled gate pack.
- Permission to record our own controlled gate clip.
- Approval to use a paid or commercial ANPR dataset.
- Written commercial permission for a request-gated academic dataset.
