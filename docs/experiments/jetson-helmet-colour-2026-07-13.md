# Jetson helmet-colour validation — 2026-07-13

## Decision

Helmet colour is a lightweight post-process on fresh helmet detections. It does
not add another neural model or another full-frame pass. Worker and rider helmet
capabilities share the Jetson's fixed-prompt TensorRT helmet profile, and colour
is classified only inside each detected helmet box.

The feature is fail-open. An empty allowed-colour list labels colours without
alerting. Uncertain crops become `unknown`, and a disallowed colour must be seen
on three fresh PPE evaluations by default before an alert is confirmed.

## TMEIC clip check

Source: `test-videos/demo-call/tmeic-pe-stores-ppe.mp4`, 60 seconds, 854 x 640.
Twelve frames were sampled at three-second intervals and sent through the live
Jetson model server's 640-pixel PPE TensorRT engine. The production-equivalent
CPU colour post-process then ran on the returned boxes.

| Measure | Result |
| --- | ---: |
| Sampled frames | 12 |
| Helmet detections | 8 |
| Yellow classifications | 8 |
| Unknown classifications | 0 |
| PPE detector median, singleton HTTP requests | 30.351 ms |
| Colour post-process median, all frames | 0.062 ms |
| Colour post-process p95, all frames | 0.865 ms |

The eight helmets appeared across four sampled frames and were classified with
colour confidence from 0.6882 to 0.8864. This is a functional clip check, not an
accuracy percentage: the clip does not provide frame-level helmet-colour ground
truth and many sampled frames contain no visible worker.

## Red/white compliance check

A labelled construction frame containing one white and one red hard hat was
sent through the same Jetson route. The detector found both helmets. The colour
post-process returned white at 0.9077 confidence and red at 0.7880 confidence.
With white configured as the only allowed colour, the white helmet was compliant
and the red helmet was a mismatch. The colour pass took 1.106 ms for both crops.

## Coverage and limitations

Synthetic unit coverage verifies white, yellow, orange, red, blue, green, and
black. Real Jetson evidence currently covers yellow, red, and white. Low light,
strong colour casts, reflective helmets, tiny boxes, and heavily occluded
helmets can return `unknown`; those results do not alert. A TMEIC acceptance set
with labelled day/night examples for every required site colour is still needed
before claiming a production accuracy percentage.

The singleton detector timing above must not be used to infer camera capacity.
The production service uses cross-camera batching and the colour pass adds only
crop-level CPU work after the already-gated PPE inference.

## Reproduce

Use `scripts/verify_helmet_colour_jetson.py` from a checkout copied to the
Jetson. Point it at the model-server container address and pass sampled JPEGs.
The command exits non-zero when a required colour is not observed and can write
the full per-frame evidence JSON with `--output`.
