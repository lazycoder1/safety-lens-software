# Jetson mobile-phone association experiment — 2026-07-12

## Decision

Increase the horizontal person-association allowance for a detected phone from
20% to 25% of person width. This is the smallest evaluated change that recovered
a phone held just outside a torso-aligned person box. It adds no model invocation,
memory use, or inference latency.

Do not add a conditional person-crop COCO pass from this experiment. The crop
pass did not recover a full-frame detection miss, added a false positive, and
would consume one inference call per selected person.

Do not add an always-on YOLOE phone pass. The first dynamic-prompt request took
5.46 seconds to warm and increased model-server resident memory from about
270 MiB to 884 MiB. A dedicated phone specialist still requires a proper
labelled benchmark before production use.

## Dataset and method

The paired set contained six phone-use positives and four negatives. Every
image was resized to both 854x480 and 352x288 to represent the two office-camera
source shapes, producing 20 labelled samples. YOLO26 Small COCO inference ran
through the live Jetson model-server JPEG endpoint at 960 input and 0.15
confidence.

The stability run repeated every sample five times, for 100 observations:

| Association policy | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current 20% horizontal padding | 35 | 0 | 25 | 40 | 100% | 58.33% |
| Candidate 25% horizontal padding | 40 | 0 | 20 | 40 | 100% | 66.67% |

All five recovered observations came from the same positive pose: a person
holding a phone beyond the side of the detected person box. Padding above 25%
did not recover another labelled sample, so the larger values were rejected.

## Conditional crop result

On the 20 paired samples, a person-crop COCO pass produced 58.3% recall and
87.5% precision. Relative to full-frame phone presence it recovered none of the
three misses and introduced one negative hit. Relative to alert-quality
association it recovered one case, but the 25% geometry change recovered that
case without extra inference or a false positive.

## Limits

This is a small, pose-diverse engineering set rather than a customer acceptance
dataset. It proves the scoped geometry change, not general mobile-phone model
accuracy. The remaining misses are detector misses or missing person context;
they should be addressed with a separately benchmarked phone-use specialist,
not by widening association geometry further.
