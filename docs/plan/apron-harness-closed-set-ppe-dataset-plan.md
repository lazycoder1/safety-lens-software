# Apron And Harness Closed-Set PPE Dataset Plan

Date: 2026-06-21

## Current State

Local YAML/runtime evidence now proves the current pilot behavior for:

- `factory_missing_apron_active`
- `factory_apron_false_positive_guard`
- `factory_apron_detector_window_suppression`
- `factory_missing_harness_active`
- `factory_harness_false_positive_guard`
- `factory_harness_detector_window_suppression`

This is not yet a production PPE compliance model. The current apron and harness path still depends on YOLOE prompt behavior, camera-specific confidence, and curated local footage. It is acceptable for scoped pilot evidence, but not for broad sales claims, Jetson 3-camera throughput claims, or commercial PPE compliance claims.

Operator gate:

```bash
.venv/bin/python scripts/apron_harness_readiness_doctor.py \
  --out qa/video_eval/results/apron_harness_readiness_doctor.json
```

This gate should pass the current pilot evidence while reporting `production_gate_passed=false`. It also records whether the capture manifest schema passes, which label counts are still below the pilot/production minimums, the next apron/harness capture batches required, whether the dry-run training plan selects MPS, CUDA, or CPU, and whether the capture matrix has approved commercial data ready for training. By default it regenerates the concise controlled-capture kickoff at `qa/video_eval/results/apron_harness_capture_kickoff.md`, the full work order at `qa/video_eval/results/apron_harness_capture_work_order.md`, `qa/video_eval/results/apron_harness_capture_matrix.csv`, and `qa/video_eval/results/apron_harness_production_capture_matrix.csv`; override those with `--capture-kickoff-out /path/to/kickoff.md`, `--capture-work-order-out /path/to/work-order.md`, `--capture-matrix-csv-out /path/to/matrix.csv`, and `--production-capture-matrix-csv-out /path/to/production-matrix.csv` when needed. The readiness JSON records the capture-manifest SHA-256 plus generated kickoff, work-order, and matrix CSV SHA-256 values, and the progress gate reconciles approved matrix rows against the manifest's `counts.labeled_images_per_class`, so staging and field capture can verify that the handoff files match the audited manifest. It must only flip production readiness after apron and harness closed-set promotion reports, the model registry report, and the `factory_ppe_3cam` Jetson full gate all point to the same candidate report SHA and trained export SHA.

## Web Search Findings

Primary and source-adjacent checks updated on 2026-06-21 and refreshed on 2026-06-22:

- Ultralytics YOLO26 docs and paper: https://docs.ultralytics.com/models/yolo26/ and https://arxiv.org/abs/2606.03748. YOLO26 is the current Ultralytics real-time family and a valid benchmark candidate for new nano/small closed-set packs. It is not an automatic production replacement until AGPL/Enterprise licensing, package compatibility, export behavior, local runtime, alert logic, and Jetson throughput are proven.
- Ultralytics export docs: https://docs.ultralytics.com/modes/export/. ONNX and TensorRT remain the preferred Jetson handoff formats.
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR. The repository is Apache-2.0 licensed and the checked latest release is v3.7.0 on 2026-06-11. PP-OCRv6 is relevant for OCR workloads such as ANPR text recognition after plate detection, but it is not an object detector for apron or safety harness compliance.
- Open Images V7 boxable classes: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv. The official class list is useful for generic objects, but it does not include apron or safety harness as closed-set classes.
- SH17 PPE dataset: https://github.com/ahmadmughees/SH17dataset and https://arxiv.org/abs/2407.04590. SH17 has useful PPE references for helmet, vest, gloves, shoes, masks, face guards, glasses, and suits, but it does not include apron or safety harness. The repository lists CC BY-NC-SA 4.0 plus educational/research-oriented terms, so it is not an automatic commercial training source.
- SFCHD-SCALE: https://github.com/lijfrank/SFCHD-SCALE and https://arxiv.org/abs/2306.02098. It covers person, safety helmet, safety clothing, other clothing, head, blurred clothing, and blurred head. It does not cover apron or safety harness and the checked repository does not provide a clear commercial training license.
- Exact GitHub repository searches for apron/harness YOLO/PPE dataset terms did not find a single credible combined apron+harness production source.
- 2026-06-22 Roboflow Universe search found CC BY 4.0 seed candidates:
  - Safety Harness Dataset by parkhm: https://universe.roboflow.com/parkhm/safety-harness-dataset. It lists 65 images and classes `top` and `safety-harness`. This is too small for production, but it is a possible harness seed source after export/provenance review.
  - kit-att-det-apron-gloves by Vaanuvaa: https://universe.roboflow.com/vaanuvaa-xtmfk/kit-att-det-apron-gloves. It lists 1,351 images and classes `apron`, `gloves`, `no_apron`, and `no_headwear`. This is a possible apron seed source after person-label and hard-negative reconciliation.
  - Apron Detection by knowledgeflex: https://universe.roboflow.com/knowledgeflex/apron-detection. It lists 576 images and class `Wearing-Apron`. This is apron-positive seed data only; it does not cover harness, lanyard, or enough negative cases.
  - PPE food manufacturing by Stock Hive: https://universe.roboflow.com/stock-hive/ppe-food-manufacturing-7n4bs. It lists 476 images and apron/mask/gloves/goggles/haircap classes. This is useful domain seed data, not a production pack by itself.
  - KitchenHygiene by Kitchen Hygiene: https://universe.roboflow.com/kitchen-hygiene-efuu5/kitchenhygiene. It lists 9,400 images with apron/no_apron plus hygiene classes. This is the strongest apron-volume candidate, but still needs export-term, provenance, split, person-box, and domain-fit review.
- 2026-06-23 Roboflow Universe refresh found two current apron/person/no-apron candidates:
  - FJWEPFJ1 by Workspace: https://universe.roboflow.com/workspace-otd88/fjwepfj1. It lists 9,694 images with numbered `00.person`, `01.apron`, and `02.no_apron` classes under CC BY 4.0. This is the strongest current apron person-box reconciliation candidate, but still needs export-term, provenance, privacy/identity, split-leakage, duplicate-lineage, numbered-class mapping, and manifest-import review.
  - eqjo by Workspace: https://universe.roboflow.com/workspace-otd88/eqjo. It lists 4,552 images with numbered `00.person`, `01.apron`, and `02.no_apron` classes under CC BY 4.0. This is a useful second current apron source, but it may overlap with FJWEPFJ1 and needs the same duplicate-lineage, export, provenance, privacy, split, and class-mapping review.
- 2026-06-23 Roboflow Universe refresh found one additional current apron-positive supplement:
  - Apron Detection by new-workspace-ndspf: https://universe.roboflow.com/new-workspace-ndspf/apron-detection-jpnm2. It lists CC BY 4.0, 576 images, one `Wearing-Apron` class, and a Roboflow 3.0 hosted model trained on 2025-11-28. Treat this as supplemental apron-positive review material only; it lacks no-apron, person, harness, and safety_lanyard coverage and does not replace closed-set YOLO26n/s training.
- Follow-up 2026-06-22 Roboflow Universe search found additional harness seed candidates:
  - Work at Height Safety by Proyecto Prevencion Predictiva: https://universe.roboflow.com/proyecto-prevencion-predictiva/work-at-height-safety. Refreshed search/page metadata shows about 12.8k visible public images and classes including harness, helmet, person, scaffolding, MEWP, and ladder, while the model API page says v3 was trained on 30,733 images. This is the strongest harness-volume candidate found so far, but it still needs export-term, provenance, class-mapping, person-box, lanyard, hard-negative, and split review.
  - harness by Harness: https://universe.roboflow.com/harness-s4xxh/harness-usugr. It lists 9,802 images and classes including Person, harness, lifeline, no harness, anchored, NotAnchored, helmet, ladder, gloves, and safety-shoe labels under CC BY 4.0. This is now a top harness/lifeline candidate, but it still needs export-term, provenance, person-box, lanyard/lifeline mapping, anchored taxonomy, split-leakage, and hard-negative review.
  - Work at Height Safety D2 by Proyecto Prevencion Predictiva: https://universe.roboflow.com/proyecto-prevencion-predictiva/work-at-height-safety-d2. It lists 2,087 images and focused `person` plus `harness` classes under CC BY 4.0. This may be easier to review for person/harness mapping than the broader source, but it still needs export-term, provenance, split-leakage, lanyard-gap, and hard-negative review.
  - Harness_Detection_V1 by PS: https://universe.roboflow.com/ps-3z4y0/harness_detection_v1. It lists 1,142 images, CC BY 4.0 licensing, and classes including person, harness, lifeline, helmet, vest, belt, no-belt, and no-front-seftybelt. It is useful as a harness/lifeline companion source if class normalization and split review pass.
  - harness_detection by Harness: https://universe.roboflow.com/harness-ptv9o/harness_detection-coo5d. It lists about 1,400 images and many classes including person, safety harness, safety-harness, safety belt, full-body-harness, hooks/ropes, vehicles, and PPE negatives. This may help hard negatives but needs class normalization before training.
  - scaffold, harness by MKS Temp: https://universe.roboflow.com/mks-temp/scaffold-harness. It lists 3,214 images and classes `human`, `safety-harness`, `Guardrail`, and `Suspension_scaffold` under CC BY 4.0. It may help scaffold/elevated-angle coverage after human-to-person mapping and split review.
  - Harness by PUBLIC PLAN: https://universe.roboflow.com/public-plan-tz9to/harness-u4vco. It lists 560 images, person/truck/safety-harness classes, CC BY 4.0 licensing, and public model-page metrics. It is supplemental because it lacks lanyard labels and is too small alone.
  - harness by yolo: https://universe.roboflow.com/yolo-cyzmy/harness-aojio. It lists only 50 images, but includes person, Harness/harness, `lanyard-with-shock-basorber`, and `lifeline`. Treat it as lanyard taxonomy review material, not production training data by itself.
  - harness by myproj: https://universe.roboflow.com/myproj-1feo9/harness-guni2. It lists 1,290 images and classes `Harness`, `Safety Vest`, and `NO-Safety Vest` under CC BY 4.0. It may help vest-vs-harness hard negatives after person-box and split review.
  - safety harness by donga highschool second grade podol: https://universe.roboflow.com/donga-highschool-second-grade-podol-33cvu/safety-harness. It lists 187 images, body/harness classes, and Public Domain licensing. It is useful seed material, not production coverage by itself.
  - full body harness by labellingapd: https://universe.roboflow.com/labellingapd/full-body-harness-cgr0q. It lists 70 images and one full-body-harness class. It is too small alone and still needs person-box and hard-negative reconciliation.

These seed sources change the sourcing status from "no public candidate found" to "public seed sources found, but unapproved and insufficient for production." They do not remove the production blockers: the capture manifest, production matrix, label counts, training provenance, side-by-side promotion, and Jetson full gate still need to pass.

A same-day alternate-source scan is recorded in `qa/video_eval/results/apron_harness_alternate_source_scan_2026_06_23.md`. Hugging Face/general-PPE sources found there are useful references for helmet, vest, shoes, mask, goggles, and gloves, but they do not include the required apron, safety_harness, or safety_lanyard classes. They do not change the production path or unblock training.

A 2026-06-24 source refresh is recorded in `qa/video_eval/results/apron_harness_source_recheck_2026_06_24.md`. The refresh rechecked the current YOLO26 production-model direction, the top Roboflow apron/harness/lanyard seed candidates, Roboflow search pages for apron/no_apron/harness/safety-harness/lifeline, and Hugging Face/general PPE pretrained shortcuts. It did not find an acceptable pretrained shortcut for the required four-class schema or any public source that can be marked `approved_for_training=true` without the existing human/legal source-review, seed-import, controlled-capture, label-review, training, promotion, registry, runtime, and Jetson gates.

The generated seed-source work order is priority-ranked from `qa/video_eval/model_packs.yaml`. Open `qa/video_eval/results/apron_harness_source_review_kickoff.md` first for the concise non-approving operator handoff, then use the generated packets and checklist for the full review. Review the best candidates first instead of treating all public hits equally: `roboflow_work_at_height_safety` for broad harness/person/work-at-height coverage, `roboflow_harness_s4xxh` for harness/lifeline/no-harness volume, `roboflow_workspace_otd88_fjwepfj1` and `roboflow_workspace_otd88_eqjo` for current apron/person/no-apron coverage, `roboflow_safety_food_system` and `roboflow_kitchen_hygiene` for apron/no-apron volume, and `roboflow_kit_att_det_apron_gloves` for focused apron/gloves mapping. Smaller or noisy sources, including `roboflow_new_workspace_apron_detection_jpnm2`, remain supplemental unless their review proves unique production coverage.

Before importing any public seed source into a capture manifest, run the seed-source review gate:

```bash
.venv/bin/python scripts/apron_harness_seed_source_doctor.py \
  --out qa/video_eval/results/apron_harness_seed_source_review.json \
  --work-order-out qa/video_eval/results/apron_harness_seed_source_review.md \
  --import-template-out qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml \
  --review-checklist-csv-out qa/video_eval/results/apron_harness_seed_source_review_checklist.csv \
  --review-evidence-template-dir qa/video_eval/results/apron_harness_seed_source_review_evidence \
  --review-packet-dir qa/video_eval/results/apron_harness_seed_source_review_packets \
  --next-review-batch-out qa/video_eval/results/apron_harness_next_source_review_batch.json \
  --review-kickoff-out qa/video_eval/results/apron_harness_source_review_kickoff.md \
  --source-coverage-plan-out qa/video_eval/results/apron_harness_source_coverage_plan.json \
  --review-bundle-out qa/video_eval/results/apron_harness_source_review_bundle.json
```

This gate is expected to be blocked until every imported source has reviewed license terms, export terms, dataset-card provenance, privacy/identity risk, class mapping, person-box coverage, hard-negative coverage, train/val/test split handling, and a manifest import plan. It also enforces `policy.source_research_max_age_days` from `qa/video_eval/model_packs.yaml`, so web-researched source entries must have a fresh `checked` date before review/import. It also writes one fillable review-evidence YAML per candidate in `qa/video_eval/results/apron_harness_seed_source_review_evidence/`, one reviewer-facing source packet per candidate in `qa/video_eval/results/apron_harness_seed_source_review_packets/`, and a non-approving coverage plan at `qa/video_eval/results/apron_harness_source_coverage_plan.json`. The coverage plan shows which candidate sources appear to cover `person`, `apron`, `safety_harness`, and `safety_lanyard` from the non-approving suggested mappings, preserves the source image-count/class metadata needed for review, and records a person-box reconciliation summary per capability. Current apron and harness candidates now both have `candidate_person_mapping_present_pending_review`; this means the person-box gap is reviewable, not approved. Production data still needs source approval, reviewed export artifacts, controlled capture, manual person-box annotation on approved seed exports, or manually reviewed auto-labeled person boxes before training. The coverage plan is a prioritization aid only, not training approval. Reviewers fill the matching evidence file, then record its path and SHA-256 in the checklist. The generated checklist CSV is only a source-review work aid; it does not approve training data by itself. A source must not be used for training unless its model-pack entry records `approval_status=approved_for_training`, `approved_for_training=true`, `manifest_import_path`, `reviewed_by`, `reviewed_at`, `review_evidence_path`, `review_evidence_sha256`, `completed_review` with every required item approved, and no remaining `blocker`. The `review_evidence_path` must point to a local review evidence file, and `review_evidence_sha256` must match that file. The file must be a YAML/JSON `apron_harness_seed_source_review_evidence` document with matching source, capability, reviewer, timestamp, and approved `review_items.<item>.evidence_ref` entries for every required review item.

Before applying a filled checklist or validating any seed import manifest, verify the generated handoff bundle and every recorded artifact hash:

```bash
.venv/bin/python scripts/apron_harness_seed_source_doctor.py \
  --review-bundle-out "" \
  --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json
```

The bundle validation is non-approving. It proves the work order, checklist, import template, next-review batch, kickoff, review packets, and review-evidence templates still match the source-review fingerprint and SHA-256 values, but it does not make any source training-usable.

After a reviewer fills the checklist, apply it to a new model-pack YAML path instead of hand-editing the source file in place:

```bash
.venv/bin/python scripts/apron_harness_seed_source_doctor.py \
  --model-packs qa/video_eval/model_packs.yaml \
  --apply-review-checklist-csv /path/to/filled/apron_harness_seed_source_review_checklist.csv \
  --updated-model-packs-out /path/to/reviewed/model_packs.yaml
```

The apply step rejects approved rows unless every review checkbox is true, `training_usable=true`, the blocker is cleared, reviewer/timestamp/import-path fields are present, and a local review evidence bundle path plus matching SHA-256 is recorded. Use the reviewed model-pack YAML in later seed-source and seed-import validation commands.

After a source is approved, fill a copy of `qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml` and validate it before adding any public seed clips to the capture manifest:

```bash
.venv/bin/python scripts/apron_harness_seed_source_doctor.py \
  --out qa/video_eval/results/apron_harness_seed_source_review.json \
  --work-order-out qa/video_eval/results/apron_harness_seed_source_review.md \
  --import-template-out qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml \
  --review-checklist-csv-out qa/video_eval/results/apron_harness_seed_source_review_checklist.csv \
  --review-evidence-template-dir qa/video_eval/results/apron_harness_seed_source_review_evidence \
  --review-packet-dir qa/video_eval/results/apron_harness_seed_source_review_packets \
  --next-review-batch-out qa/video_eval/results/apron_harness_next_source_review_batch.json \
  --review-kickoff-out qa/video_eval/results/apron_harness_source_review_kickoff.md \
  --source-coverage-plan-out qa/video_eval/results/apron_harness_source_coverage_plan.json \
  --review-bundle-out qa/video_eval/results/apron_harness_source_review_bundle.json \
  --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json \
  --validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml
```

The import manifest must keep `include_in_training=false` until the source review is `training_usable`, the filled import entry has matching `source_review_sha256`, `review_status=approved_for_training`, `reviewed_by`, `reviewed_at`, matching `manifest_import_path`, remote immutable `raw_export_ref` such as object storage or HTTPS export artifact URI, `raw_export_sha256`, `export_format=yolo`, completed review approvals, class mapping into the local `person` plus target-PPE taxonomy, person-box policy, hard-negative policy, explicit train/val/test split handling, and nonzero expected counts for `person` plus the target PPE class. Generated import rows include `review_artifacts` pointing to the source's review packet and review-evidence template plus `approval_preconditions`; these are reviewer navigation aids only, not approval. When the seed-source review has generated packets/templates, validation also requires the filled import row to preserve the matching review-artifact paths and SHA-256 values and verifies the files still hash to those values. Local filesystem paths, repo paths, and workstation scratch paths are not acceptable `raw_export_ref` values, and the reviewed export must have a 64-character SHA-256 digest. The `source_review_sha256` is a stable fingerprint of the source-review decisions with generated timestamps ignored; regenerate the import manifest when approvals or candidate metadata change. This prevents a web-found dataset from becoming training data without a reviewed import contract.

Do not use paid, gated, academic-only, customer-private, or identifiable-person footage without explicit approval.

## Model Selection Decision

Production PPE should move to a commercial-safe closed-set detector trained for our exact classes:

- Production candidates to benchmark now: `yolo26n` or `yolo26s`, because YOLO26 is the current Ultralytics real-time family and reports better deployment efficiency.
- Older YOLO11/YOLOv8 nano/small models are legacy/runtime baselines only. They are not accepted by the apron/harness production trainer unless a future explicit exception is added with license, runtime, accuracy, and Jetson evidence.
- Current pilot only: YOLOE/YOLOE-26 open-vocabulary models, because prompt stability, false positives, licensing, and Jetson 3-camera throughput are not yet proven.

The production choice is whichever candidate passes license, export, local Mac, YAML/runtime, false-positive, and Jetson 3-camera gates. "Latest" is a filter for candidates, not a reason to skip evidence.

PP-OCRv6 is not a PPE model. Keep PaddleOCR PP-OCRv6 tiny/small for ANPR OCR, not for apron or harness detection.

## Required Footage

Minimum controlled capture pack:

- 300 to 500 labeled images per target class for the first pilot.
- 1,000 or more labeled images per target class before production confidence.
- At least 3 camera angles: front, side, and elevated CCTV-style.
- At least 3 distance bands: close, medium shop-floor, and wide surveillance.
- Day/bright indoor, dim indoor, backlit, glare, and motion-blur examples.
- Occlusion examples: partial torso, blocked waist, overlapping workers, tools or machinery in front.
- Active-window and inactive-window clips can reuse the same footage, but must run through separate YAML scenarios.

Apron positives:

- Denim apron.
- Work apron.
- Kitchen or food-service apron.
- Protective industrial apron.
- Partial apron visible from the side.

Apron hard negatives:

- Safety vest.
- Jacket.
- Lab coat.
- Shirt with color block.
- Tool belt.
- Loose cloth or scarf near the torso.

Harness positives:

- Full-body safety harness.
- Fall-arrest harness.
- Visible lanyard or tether.
- Harness over safety vest.
- Harness partially hidden by tools or movement.

Harness hard negatives:

- Backpack straps.
- Tool belts.
- Seat belts.
- Ropes, cables, slings, or hoses.
- Reflective vest stripes.

Privacy and legal requirements:

- Capture or use footage only with permission.
- Store raw videos outside the repo if people are identifiable.
- Commit only metadata, redacted frame checks, manifests, and result summaries.
- If customer footage is required, get approval before copying it into any local or cloud workflow.

## Annotation Schema

Use object boxes, not missing-PPE labels. Missing PPE should be derived by runtime association between a `person` and visible required PPE.

Classes:

```yaml
names:
  0: person
  1: apron
  2: safety_harness
  3: safety_lanyard
```

Each listed label file must also carry manual-review metadata in `yolo_labels`. Auto-labeling is allowed only before this review step.

```yaml
yolo_labels:
  - path: labels/frame_0001.txt
    review_status: approved
    reviewer: qa_reviewer_or_vendor_id
    reviewed_at: 2026-06-21T00:00:00+00:00
    source_clip_id: factory_harness_side_001
    split: train
```

Label rules:

- Use tight boxes around visible object boundaries.
- Label partial objects only when at least 40 percent is visible and the class is unambiguous.
- Mark ambiguous examples as ignore/excluded, not as negatives.
- Include `person` boxes for every worker in the frame.
- Label `safety_lanyard` separately when visible; runtime can treat harness or lanyard evidence as satisfying the harness rule only after validation.

Per-clip metadata:

```yaml
clip_id: factory_harness_side_001
source: controlled_capture
permission: internal_cleared
camera_angle: side
distance_band: medium
lighting: indoor_bright
motion_blur: low
target_capabilities:
  - harness_required
expected_visible_classes:
  - person
  - safety_harness
  - safety_lanyard
positive_variant_tags:
  - full_body_safety_harness
  - visible_lanyard_or_tether
hard_negative_tags:
  - backpack_straps
notes: harness visible over vest, lanyard visible on right side
```

Capture-pack validation:

```bash
.venv/bin/python scripts/apron_harness_dataset_doctor.py \
  --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml \
  --schema-only
```

The schema-only template run should also emit a capture deficit. With the checked-in template, the current pilot deficit is 1,200 missing label annotations across `person`, `apron`, `safety_harness`, and `safety_lanyard`, plus two next capture batches: `apron_required_closed_set_capture` and `harness_required_closed_set_capture`. The generated work order includes a per-variant capture matrix so field capture can record apron positives, harness/lanyard positives, and hard negatives separately.

Strict validation does not trust declared label counts or coverage summaries by themselves. When `--schema-only` is not used, `scripts/apron_harness_dataset_doctor.py` reads every listed YOLO label file, requires `review_status=approved`, `reviewer`, `reviewed_at`, `split`, and a `source_clip_id` that appears in `clips`, counts the distinct label files containing each required class, rejects duplicate label paths, requires reviewed labels for every required class in both `train` and `val`, requires held-out `test` split coverage for every required class in production mode, requires at least 10 percent of the production per-class target in both `val` and `test`, rejects the same `source_clip_id` appearing in more than one dataset split, and fails if `counts.labeled_images_per_class` claims more reviewed class images than the listed labels prove. It also requires every top-level `coverage` value to be backed by clip metadata: camera angle, distance, lighting, motion blur, `positive_variant_tags`, and `hard_negative_tags`. Public or commercial-dataset seed clips must set `source_ref`, pass `--seed-source-review-report`, and pass `--seed-import-manifest`; the referenced seed-source candidate must be `training_usable=true`, `approval_status=approved_for_training`, include `manifest_import_path`, `reviewed_by`, `reviewed_at`, `review_evidence_path`, `review_evidence_sha256`, and match the clip's apron/harness target capability, while the seed-import manifest must include a matching source-review fingerprint plus a matching source/capability entry with `include_in_training=true`, `review_status=approved_for_training`, completed review approvals, class mapping into the local `person` plus target-PPE taxonomy, matching `manifest_import_path`, remote immutable `raw_export_ref`, `raw_export_sha256`, `export_format=yolo`, person-box policy, hard-negative policy, explicit train/val/test split handling, and nonzero expected counts for `person` plus the target PPE class. With the current 1,000-image production target, that means at least 100 reviewed validation images and 100 reviewed held-out test images per required class. A production manifest therefore needs accurate count metadata, backed coverage metadata, approved seed-source provenance and approved seed-import metadata where applicable, and the corresponding approved label-file list.

Generate the operator work order from that same deficit:

```bash
.venv/bin/python scripts/apron_harness_dataset_doctor.py \
  --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml \
  --mode pilot \
  --schema-only \
  --emit-capture-work-order qa/video_eval/results/apron_harness_capture_work_order.md \
  --emit-capture-matrix-csv qa/video_eval/results/apron_harness_capture_matrix.csv \
  --emit-label-review-csv qa/video_eval/results/apron_harness_label_review_template.csv
```

Generate the production capture matrix separately:

```bash
.venv/bin/python scripts/apron_harness_dataset_doctor.py \
  --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml \
  --mode production \
  --schema-only \
  --emit-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv \
  --emit-label-review-csv qa/video_eval/results/apron_harness_production_label_review_template.csv
```

The capture matrix CSV includes fillable progress fields: `captured_examples`, `labeled_examples`, `review_status`, `permission`, `raw_storage_ref`, `owner`, `due_date`, and `status_notes`. A row only clears the progress gate when `labeled_examples` is at least `recommended_examples`, `review_status=approved`, `permission` is one of the commercial-cleared values accepted by the manifest doctor, and `raw_storage_ref` points outside the repo. The label-review CSV expands that matrix into one planned YOLO label-file row per recommended example with a suggested train/val/test split. After review, import the approved rows into a YAML capture manifest instead of editing `yolo_labels` by hand:

```bash
.venv/bin/python scripts/apron_harness_dataset_doctor.py \
  --manifest /path/to/cleared/apron_harness_capture_manifest.yaml \
  --mode production \
  --import-label-review-csv /path/to/filled/apron_harness_label_review.csv \
  --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml
```

The importer converts only `review_status=approved` rows into `yolo_labels`, creates `clips` only from filled reviewed row metadata for new `source_clip_id` values, rejects approved rows with uncleared permissions or repo-local `raw_storage_ref`, requires `reviewer`, `reviewed_at`, `source_clip_id`, split metadata, and clip metadata when the clip is not already listed, and recomputes `counts.labeled_images_per_class` from the referenced YOLO label files. Use `--mode production` when generating the reviewed manifest for production training, so the sidecar proves production split/holdout validation instead of pilot-only validation. It does not approve rows, invent absent clip metadata, or bypass missing label files. When `--manifest` is provided, the gate also checks that approved matrix totals reconcile with the manifest's labeled-image counts per class. Validate a filled production matrix before production training:

The same import command also writes `/path/to/cleared/apron_harness_capture_manifest.reviewed.yaml.label_review_import.json`. Keep that sidecar with the reviewed manifest. It records the source-manifest SHA-256, filled label-review CSV SHA-256, reviewed-manifest SHA-256, imported/skipped row counts, imported clip counts, recomputed class counts, strict `updated_manifest_validation`, and the training-gate assertions that review metadata, reviewed clip metadata, cleared permissions, non-repo raw storage, reviewed-manifest validation, and recomputed counts were enforced.

If a public seed source and its raw YOLO export ZIP are approved, materialize it through the dataset doctor instead of manually copying labels. This path validates the seed-source review report, the filled seed-import manifest, the local ZIP SHA-256, and the YOLO export preflight, including `data.yaml` classes, mapped source labels, train/valid/test image and label presence, matching image files for every label file, and required local-class label-file counts; remaps source classes into the local `person`/`apron`/`safety_harness`/`safety_lanyard` taxonomy; writes local `images/` and `labels/`; adds reviewed capture-manifest clips; recomputes counts from converted labels; and writes `/path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml.seed_export_import.json`. The camera-angle, distance, lighting, motion-blur, positive-variant, and hard-negative flags must be reviewed metadata for the imported source:

```bash
.venv/bin/python scripts/apron_harness_dataset_doctor.py \
  --manifest /path/to/cleared/apron_harness_capture_manifest.yaml \
  --mode production \
  --schema-only \
  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \
  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \
  --import-approved-seed-exports \
  --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml \
  --seed-import-camera-angle front \
  --seed-import-distance-band medium \
  --seed-import-lighting indoor_bright \
  --seed-import-motion-blur low \
  --seed-import-positive-variant-tags "full_body_safety_harness;visible_lanyard_or_tether"
```

Production training, candidate promotion, side-by-side promotion, model-registry copy, and readiness gates reject public/commercial seed clips when this `.seed_export_import.json` sidecar is missing, stale, not production-validated, not tied to the same seed-source review, seed-import manifest, and capture-manifest SHA, or missing the preserved YOLO export preflight evidence for exact ZIP SHA, zero orphan labels, and required local-class label counts.

```bash
.venv/bin/python scripts/apron_harness_dataset_doctor.py \
  --manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml \
  --mode production \
  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \
  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \
  --validate-capture-matrix-csv /path/to/filled/apron_harness_production_capture_matrix.csv
```

When real cleared footage is available, copy the template to the dataset workspace, replace the metadata and label paths, then run without `--schema-only`. Use `--mode production` before claiming production readiness.

After the capture manifest passes strict validation, generate the YOLO training config:

```bash
.venv/bin/python scripts/apron_harness_dataset_doctor.py \
  --manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml \
  --mode production \
  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \
  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \
  --emit-yolo-dataset-yaml /path/to/cleared/dataset.yaml
```

Before training, generate a dry-run training plan:

```bash
.venv/bin/python scripts/apron_harness_train.py \
  --data /path/to/cleared/dataset.yaml \
  --capture-manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml \
  --capture-matrix-csv /path/to/filled/apron_harness_production_capture_matrix.csv \
  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \
  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \
  --capture-preflight-mode production \
  --require-capture-preflight \
  --model yolo26n.pt \
  --device auto \
  --out-plan /path/to/cleared/apron_harness_training_plan.json
```

With `--require-capture-preflight`, the trainer also validates the `rakshak_lens` block inside `dataset.yaml`: `source_manifest` must resolve to the same reviewed manifest passed through `--capture-manifest`, `source_manifest_sha256` must match that exact reviewed capture manifest, the permission must be commercially cleared, and `missing_ppe_label_policy` must remain `derive_missing_ppe_from_person_to_visible_ppe_association`. It also requires both production handoff sidecars: `/path/to/filled/apron_harness_production_capture_matrix.csv.manifest.json` for the capture matrix and `/path/to/cleared/apron_harness_capture_manifest.reviewed.yaml.label_review_import.json` for the filled label-review CSV import. Finally, it runs the capture manifest schema/provenance review with `--seed-source-review-report` and `--seed-import-manifest`, so public/commercial seed clips cannot enter training unless both the source-specific review and filled import approval have already passed.

The readiness doctor records the same strict trainer dry-run under `closed_set_handoff.production_training_plan_preflight`. This block must show `checked=true` and `ok=true` before real training should be attempted. With the checked-in template it is expected to fail because the production capture matrix, reviewed labels, and label-review import sidecar are incomplete; that failure keeps the production claim blocked while still proving the handoff command path is wired.

The checked-in example can be used to verify command wiring only:

```bash
.venv/bin/python scripts/apron_harness_train.py \
  --data qa/video_eval/datasets/apron_harness_dataset.example.yaml \
  --model yolo26n.pt \
  --device cpu \
  --out-plan /tmp/apron_harness_training_plan.json
```

## Training Plan

1. Extract frames from cleared videos into a dataset workspace outside committed source.
2. Create `dataset/apron_harness_ppe/dataset.yaml` with the class schema above.
3. Use auto-labeling only as an accelerator; every apron and harness/lanyard label must be manually reviewed and listed with approved review metadata before training.
4. Run the dry-run plan command above and confirm it selects a nano/small closed-set model, never YOLOE. The real dry-run and `--execute` commands must include the capture manifest and filled matrix CSV, and the dataset YAML must point back to that same capture manifest so the training script can enforce provenance before model training starts.
5. Train `yolo26n` first if the local package and license path are approved. Train `yolo26s` only if `n` misses too many clear examples and the Jetson budget still passes.

```bash
.venv/bin/python scripts/apron_harness_train.py \
  --data /path/to/cleared/dataset.yaml \
  --capture-manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml \
  --capture-matrix-csv /path/to/filled/apron_harness_production_capture_matrix.csv \
  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \
  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \
  --capture-preflight-mode production \
  --require-capture-preflight \
  --model yolo26n.pt \
  --device auto \
  --epochs 100 \
  --batch 8 \
  --export-format onnx \
  --out-plan /path/to/cleared/apron_harness_yolo26n_result.json \
  --execute
```

6. If both `yolo26n` and `yolo26s` fail package/runtime/license or quality gates, keep apron/harness production blocked rather than falling back to older model families.
7. Gate the trained/exported candidate before runtime promotion:
   - `scripts/apron_harness_candidate_doctor.py` and `scripts/apron_harness_promotion_doctor.py` must carry the candidate `label_review_import_manifest`; promotion reports without that sidecar are not production evidence.
   - `scripts/apron_harness_promotion_doctor.py` records `candidate_report_sha256`; readiness rejects apron promotion, harness promotion, and model registry reports unless they share the same candidate-report SHA and trained export SHA.
   - The `.label_review_import.json` sidecar must also carry `updated_manifest_validation` from a strict, non-schema-only validation of the reviewed capture manifest. Production preflight and promotion reject sidecars where that validation is missing, failed, schema-only, non-production, or tied to a different manifest SHA.

```bash
.venv/bin/python scripts/apron_harness_candidate_doctor.py \
  --training-result /path/to/cleared/apron_harness_yolo26n_result.json \
  --out /path/to/cleared/apron_harness_candidate_report.json
```

The candidate doctor requires `status: trained`, a passed production-mode `capture_preflight` from `scripts/apron_harness_train.py`, matching `dataset_provenance` and `source_lineage` from the training result, an allowed nano/small model family, the exact four-class schema, training-script-emitted `per_class_metrics` with per-class `mAP50 >= 0.75`, per-class `recall >= 0.85`, and at least one existing ONNX or TensorRT export artifact. Do not hand-edit missing per-class metrics into the result JSON; rerun training/validation if the trainer cannot extract them. Its promotion manifest also emits `training_capture_preflight`, `training_dataset_provenance`, `training_source_lineage`, `runtime_handoff.planned_model_key`, `runtime_handoff.planned_registry_path`, `runtime_handoff.selected_export.sha256`, and a `runtime_handoff.registry_entry` so the later registry step uses the same handoff contract as `qa/video_eval/model_packs.yaml` and can verify the copied artifact before activation. The side-by-side promotion doctor carries this forward as `candidate_training_source_lineage`, and the readiness doctor rejects missing or manifest-mismatched lineage before any production gate can pass.
9. Export ONNX and TensorRT candidates for staging.
10. Register the model as a closed-set PPE specialist only after it passes side-by-side tests against the current YOLOE pilot path.

## Runtime Handoff Contract

The current default runtime model family for apron and harness remains `ppe_specialist`, which is the YOLOE pilot path. `backend/model_manager.py` may expose `ppe_closed_set_candidate` as a dormant manual-install slot so candidate reports and operators have a stable handoff target, but the missing artifact must keep it `not_downloaded`. Do not add an empty artifact to `factory_ppe_3cam.registry_models`, add the key to `factory_ppe_3cam.model_keys`, or make the closed-set model the default route until the trained candidate exists and has passed the candidate, side-by-side, and Jetson gates.

Scoped side-by-side YAML hook:

```yaml
cameras:
  eval_factory_apron_closed_set:
    capabilities: [apron_required]
    safety_rule_ids: [ppe_apron]
    capability_model_overrides:
      apron_required: ppe_closed_set_candidate
```

Only `apron_required` and `harness_required` may target `ppe_closed_set_candidate`, and this override is for candidate runtime evidence only. It must be used after the registry doctor has copied a real ONNX artifact to `models/ppe_closed_set_candidate/apron-harness-ppe.onnx`; it is not production activation and must not replace the existing YOLOE pilot scenarios.

Candidate YAML files are staged but blocked until the artifact exists:

- `qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml`
- `qa/video_eval/focused/factory_apron_false_positive_guard_closed_set.yaml`
- `qa/video_eval/focused/factory_apron_detector_window_suppression_closed_set.yaml`
- `qa/video_eval/focused/factory_missing_harness_active_closed_set.yaml`
- `qa/video_eval/focused/factory_harness_false_positive_guard_closed_set.yaml`
- `qa/video_eval/focused/factory_harness_detector_window_suppression_closed_set.yaml`

The matching manifest scenarios are intentionally marked `blocked_pending_model_artifact` and are not part of the current sales-readiness coverage list. They become promotion inputs only after the registry copy report proves the real ONNX artifact is installed.

Planned production handoff key:

```yaml
planned_model_key: ppe_closed_set_candidate
planned_registry_path: models/ppe_closed_set_candidate/apron-harness-ppe.onnx
```

Promotion order:

1. Pass dataset validation and emit the real training YAML.
2. Train a nano/small closed-set model on cleared data.
3. Pass `scripts/apron_harness_candidate_doctor.py`, including the production-mode capture preflight embedded in the training result.
4. Run side-by-side YAML scenarios against the current YOLOE pilot by setting `capability_model_overrides.apron_required` or `capability_model_overrides.harness_required` to `ppe_closed_set_candidate` on candidate-only YAML scenarios.
5. Copy the trained ONNX artifact to the dormant model-manager registry path through `scripts/apron_harness_model_registry_doctor.py` and verify its SHA-256 only after both side-by-side promotion reports pass for the same candidate report and selected-export SHA.
6. Only after side-by-side and Jetson gates pass, decide whether to make the closed-set path the default apron/harness route.
7. Keep YOLOE as a pilot fallback until Jetson one-camera and 3-camera soak evidence is recorded.

Registry dry-run:

```bash
.venv/bin/python scripts/apron_harness_model_registry_doctor.py \
  --candidate-report /path/to/cleared/apron_harness_candidate_report.json \
  --out /path/to/cleared/apron_harness_model_registry_report.json
```

Registry copy after the dry-run passes:

```bash
.venv/bin/python scripts/apron_harness_model_registry_doctor.py \
  --candidate-report /path/to/cleared/apron_harness_candidate_report.json \
  --apron-promotion-report /path/to/cleared/apron_closed_set_promotion_report.json \
  --harness-promotion-report /path/to/cleared/harness_closed_set_promotion_report.json \
  --copy \
  --out /path/to/cleared/apron_harness_model_registry_report.json
```

The registry doctor refuses a candidate report that is not `ok=true`, verifies `selected_export.sha256` against the export file on disk, verifies the dormant `backend/model_manager.py` definition, and in `--copy` mode requires both apron and harness side-by-side promotion reports to be `ready_for_runtime_registration` for the same candidate-report SHA and selected-export SHA. Only then does it write the artifact to `models/ppe_closed_set_candidate/apron-harness-ppe.onnx`. A registered artifact also needs the adjacent provenance sidecar `models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json`, which records the candidate-report SHA-256, source-export SHA-256, artifact SHA-256, model key, and registry path. If the ONNX file already exists but that sidecar is missing or stale, rerun the registry doctor with `--copy`; a dry run will stay `ready_to_copy`, not `registered`. The current dormant registry path is ONNX-only; a TensorRT `.engine` export should be benchmarked by the Jetson gate and must not be copied into the `.onnx` runtime slot unless a separate engine-specific runtime path is added.

Side-by-side promotion gate:

```bash
.venv/bin/python scripts/apron_harness_promotion_doctor.py \
  --capability apron_required \
  --candidate-report /path/to/cleared/apron_harness_candidate_report.json \
  --baseline-active qa/video_eval/results/factory_missing_apron_active.json \
  --baseline-guard qa/video_eval/results/factory_apron_false_positive_guard.json \
  --baseline-suppression qa/video_eval/results/factory_apron_detector_window_suppression.json \
  --candidate-active /path/to/cleared/factory_missing_apron_active_closed_set.json \
  --candidate-guard /path/to/cleared/factory_apron_false_positive_guard_closed_set.json \
  --candidate-suppression /path/to/cleared/factory_apron_detector_window_suppression_closed_set.json \
  --out /path/to/cleared/apron_closed_set_promotion_report.json
```

Repeat the same command with `--capability harness_required` and the harness baseline/candidate result files. This gate requires the candidate-doctor report, active-window alert evidence, visible-PPE false-positive guard evidence, positive model-invocation telemetry for both baseline and candidate active/guard runs, detector-off suppression with zero candidate model invocations, and fresh screenshots before runtime registration.

Jetson promotion gate:

```bash
.venv/bin/python scripts/jetson_benchmark_doctor.py \
  --pack factory_ppe_3cam \
  --model apron-harness-ppe.onnx \
  --candidate-report /path/to/cleared/apron_harness_candidate_report.json \
  --raw-benchmark /path/to/cleared/factory_ppe_raw_benchmark.json \
  --soak-report /path/to/cleared/factory_ppe_3cam_soak.json \
  --require-full-gate \
  --out /path/to/cleared/factory_ppe_jetson_gate.json
```

Raw latency alone is not enough for production. The doctor only reports `jetson_gate_passed` when the raw CUDA benchmark and the three-camera soak report both satisfy the resource limits in `qa/video_eval/model_packs.yaml`.

Initial target metrics:

- Per-class mAP50 >= 0.75 for pilot.
- Per-class recall >= 0.85 on clear controlled footage.
- Zero missing-PPE alerts on visible-PPE false-positive guard clips.
- Zero apron/harness candidates and zero PPE model invocations in inactive detector windows.
- Jetson 3-camera soak meets the `factory_ppe_3cam` resource limits in `qa/video_eval/model_packs.yaml`.

## YAML Scenario Plan

Use one detection at a time:

1. Create a closed-set apron active YAML that enables only `apron_required`.
2. Run a missing-apron positive clip during an active detector window.
3. Run a visible-apron false-positive guard clip during an active detector window.
4. Run the same positive clip outside the capability active window and require zero apron candidates plus zero PPE model invocations.
5. Restore config.
6. Repeat the same sequence for `harness_required`.
7. Only after isolated tests pass, run a bundled `factory_ppe_3cam` staging scenario.

Do not directly edit `backend/config.json`, DB rows, or browser state to make the tests pass. If YAML cannot express a model, rule, camera, schedule, threshold, or output, extend YAML/runtime support first.

## Pass Gates

Local software gate:

- YAML validates and applies through `scripts/safetylens_site.py`.
- `scripts/video_eval.py` records active, false-positive guard, and detector-window suppression evidence.
- Reports show class counts, model invocations, alerts, policy decisions, and delivery results.
- Local Mac MPS is used when available; CPU fallback proves wiring only.

Staging Jetson gate:

- Closed-set model exported to ONNX/TensorRT candidate format.
- One-camera benchmark passes first.
- Three-camera soak records FPS, p95 latency, RAM, GPU utilization, dropped frames, stream restarts, alert counts, and false positives.
- Detector-off scheduling proves suppressed capability counters, zero model invocations, and zero emitted candidates outside active windows.

Sales gate:

- Customer-site camera angle, lighting, and PPE variants have representative evidence.
- Clear legal basis exists for all training/validation footage.
- Sales readiness report marks the exact supported scope and keeps unsupported variants blocked.

## Open Ask

To move apron/harness from pilot to sellable production, we need one of:

- Permission to capture controlled internal apron and harness footage.
- Cleared customer factory footage that matches the minimum capture pack.
- Approval to purchase or subscribe to a commercial PPE dataset.
- Written commercial permission for any gated or academic source.
- Annotation capacity for manual label review before training.
