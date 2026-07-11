# Marketing Feature Model Enablement Goal

## Paste Ready Goal Prompt

Enable the Rakshak Lens marketing feature set with real runnable models and evidence. Work locally first on this Mac using Apple Silicon MPS, then prepare the same path for a staging edge target that can run up to 3 cameras on Jetson Nano-class hardware. Use open-source models only unless a required feature has no credible public model or dataset; in that case use online web search and browser automation to find legal datasets, model weights, papers, or training sources, and ask me before using paid subscriptions, gated datasets, or paid labeling services.

Start from the brochure claims in `marketing/suite/final-send`, `qa/video_eval/manifest.yaml`, `qa/video_eval/SALES_READINESS_REPORT.md`, and `qa/video_eval/DETECTION_GOALS_AND_JETSON_BENCHMARK.md`. For every claim, map feature -> required capability -> model family -> dataset/source -> training or download path -> YAML config -> runtime test -> sales status. Use only YAML/config-driven changes for camera/rule/runtime setup. Add code only when the system lacks model registration, install, telemetry, inference, or UI needed to prove the feature.

The goal is not to make every brochure claim pass by relaxing thresholds. The goal is to make each claim honestly sellable, demo-only, blocked, or needs-work with evidence. Run one detection feature at a time: turn on only that detection, test all requirements for it, then turn it off before moving to the next detection. For every enabled detection, test daily/weekly time windows where the detection should work and windows where it should not work. Important nuance: the current UI/policy work supports alert/policy scheduling, not guaranteed detector-off scheduling. If the requirement is "detection itself should not run outside this shift," add capability-level active windows in YAML/runtime plus telemetry proving suppressed detections. Prefer nano/small YOLO, pose, OCR, or lightweight specialist models that can plausibly run 3 cameras on Jetson Nano-class hardware. Avoid YOLOE as the production path unless it is explicitly marked demo-only and benchmarked.

Acceptance: local MPS tests pass for each enabled model path, the staging plan specifies Jetson model format/FPS/resource limits, `scripts/safetylens_site.py` can validate/apply the needed YAML, `scripts/video_eval.py` records scenario evidence, sales readiness docs are updated, and unsupported marketing claims remain blocked. Alert/policy scheduling alone is not enough for "detection off outside shift" claims; those require capability-level active windows in YAML/runtime plus telemetry showing the capability was skipped or suppressed before detection candidates were emitted.

## Objective

Turn the current marketing feature list into a real model enablement program: source or train lightweight models, wire them into the existing runtime, test them locally on M1/MPS, and prepare a staging path for up to 3 camera streams on Jetson Nano-class edge hardware.

The end state is a feature matrix where every brochure claim has one of these states:

- `ready_to_sell`: model, config, runtime, UI, alert, and QA evidence are complete.
- `pilot_ready`: works locally or on staging with clear scope limits and needs customer footage before broad sales.
- `demo_only`: works on curated footage but should not be sold as production reliable.
- `needs_work`: implementation or model quality is insufficient.
- `blocked`: missing legal data, missing hardware, missing paid subscription, or claim should not be made.

## Hardware Scope

Local development target:

- Mac with Apple Silicon integrated GPU.
- Use PyTorch MPS where supported.
- Use local file-backed videos and simulated RTSP when needed.
- Local results prove software wiring and first-pass model quality, not Jetson throughput.

Staging edge target:

- Jetson Nano-class deployment goal: up to 3 cameras.
- If the actual device is original Jetson Nano 4GB, target low FPS and nano models only.
- If the actual device is Jetson Orin Nano / Orin NX, target higher FPS but still benchmark before claiming capacity.
- Production path should prefer TensorRT/ONNX-exportable nano/small models.
- Do not make 3-camera claims until staging benchmark records FPS, latency, GPU/RAM load, and alert stability.

## Source Documents

Read these first:

- `marketing/suite/final-send/`
- `marketing/suite/README.md`
- `qa/video_eval/manifest.yaml`
- `qa/video_eval/SALES_READINESS_REPORT.md`
- `qa/video_eval/DETECTION_GOALS_AND_JETSON_BENCHMARK.md`
- `docs/plan/apron-harness-closed-set-ppe-dataset-plan.md`
- `backend/capability_registry.py`
- `backend/model_manager.py`
- `backend/video_processing.py`
- `scripts/safetylens_site.py`
- `scripts/video_eval.py`

## Kickoff Findings - 2026-06-19

- Detector-off scheduling is now explicitly in scope for this goal. Existing alert/policy schedules only suppress notifications; sellable "detection is off outside shift" claims need capability active windows in YAML/runtime, runtime skips before model/candidate emission, and telemetry showing skipped model/capability work with zero emitted candidates.
- Local Apple Silicon MPS remains a hard acceptance gate for production-grade local model-pack evidence. The current 2026-06-23 probe now reports torch 2.10.0 with MPS available and the MPS tensor probe passing, so refreshed local runs can satisfy the Mac MPS gate. Jetson Nano-class three-camera throughput remains unproven until target hardware benchmarks pass.
- Hospitals/RL-M are skipped for the current validation cycle. Hospital, clinical, patient-monitoring, fall/person-down, exercise/therapy false-positive, and pose-fall model work should remain archived/backlog unless explicitly re-scoped. Current sales-readiness reporting excludes hospital scenarios through `coverage_boundaries.skipped_verticals` and `coverage_boundaries.skipped_scenario_ids` in `qa/video_eval/manifest.yaml`.
- Existing YOLOE PPE frame probes show apron and harness are mapping/confidence problems before they are dataset problems. Current `apron` prompt missed the visible-apron frame, while expanded prompts detected `denim apron` at low confidence around `0.10-0.14`. Harness prompts detected `safety harness` / `fall arrest harness` on existing frame checks around `0.10-0.20`.
- First implementation target: align apron/harness prompt mappings with the manifest/YAML safety rules, add missing `ppe_harness` default rule support, and allow YAML camera-level PPE detector confidence overrides. Do not mark apron or harness ready until full `scripts/video_eval.py` scenarios pass with class telemetry and no forbidden missing-PPE alerts.

## Current Feature Priorities

Priority 0: keep already-proven features working.

- Person detection.
- Vehicle presence.
- Mobile phone.
- Zone intrusion.
- Queue/crowd/occupancy logic from person detections.
- Object dwell/removal where COCO classes are sufficient.
- Per-camera event channels, schedules, severity, priority, messages, and cooldown.

Priority 1: make factory PPE sellable beyond helmet.

- Helmet.
- Vest.
- Gloves.
- Face mask.
- Face shield.
- Hairnet.
- Goggles.
- Boots.
- Apron.
- Harness.

Current report now has helmet, apron, gloves, boots, goggles, face mask, face shield, hairnet, and harness covered by focused active-window, false-positive guard, and detector-window suppression scenarios. Keep broad PPE as `pilot_ready`, not full production compliance, until customer-site camera angles, a commercial-safe closed-set PPE model path, and Jetson 3-camera throughput are benchmarked.

Priority 2: specialist safety features.

- Fire/smoke with a small specialist model.
- False-positive guard clips for fire/smoke.

Current fire/smoke status: `fire_smoke_public_commons`, `fire_smoke_welding_false_positive_guard`, and `fire_smoke_detector_window_suppression` now pass at the YAML-configured `0.70` fire/smoke confidence operating point. Lower thresholds (`0.35` and `0.55`) produced welding false positives, so `0.70` is the current scoped local operating point. This proves active-window P1 alerting on one public Commons fire/smoke clip, suppression on one welding/hot-work negative-control clip, and detector-off scheduling with zero fire/smoke model invocations outside the active window. Do not claim certified fire-alarm compliance, smoke-only lower-contrast events, all welding/glare/hot-work robustness, thermal-camera detection, alarm-panel integration, or Jetson 3-camera throughput until those scenarios pass separately.

Fall/person-down is no longer in the active validation scope because RL-M hospitals are skipped. Keep the historical hospital scenarios and pose-fall pack definitions for archive/backlog reference only; do not spend this cycle on pose-fall promotion, exercise/therapy false-positive suppression, or clinical patient-monitoring claims.

Priority 3: access/gate features.

- ANPR with plate detector + OCR. Use Baidu/PaddleOCR PP-OCRv6 tiny as the default recognizer path, with PP-OCRv6 small as the next tier if tiny misses too many Indian plate reads. Local Mac evidence now proves PaddleOCR 3.7.0 + PP-OCRv6 tiny on the synthetic gate clip, a public-domain still-image plate fixture, active no-plate false-positive guard, and detector-window suppression; Jetson throughput and real gate footage are still pending.
- Indian plate day/night/motion-blur validation.
- Face recognition remains blocked for production claims until consent, enrollment, accuracy, and privacy workflow are validated.

Priority 4: keep blocked unless explicitly funded and validated.

- Aggression/fight detection.
- Patient wandering or clinical patient monitoring.
- Tailgating as identity/access-control logic.
- Semantic AI search.
- Owner association for unattended objects.

## Model Policy

Model selection must use current web research before locking a runtime path. Refresh the source check during the goal, and treat any stale or unsourced recommendation as a candidate only. For every model family, check primary sources first: official docs, official GitHub releases, model cards, papers, and license files. Record the source URL, release/version date, license, supported export formats, expected hardware, and why the model was accepted or rejected.

Current-generation constraint:

- New production model packs must use 2025/2026-era candidates only.
- Current allowed production candidates are YOLO26 nano/small closed-set detectors, YOLO26 nano pose where pose is required, PP-OCRv6 tiny/small for OCR/ANPR, and other 2025/2026 candidates only after primary-source and license review.
- YOLOE-26 is allowed as a demo, pilot, source-discovery, or side-by-side benchmark path; it is not the default Jetson Nano-class production PPE detector until prompt stability, licensing, false-positive behavior, active-window telemetry, and Jetson throughput pass.
- YOLO11, YOLOv8, and YOLOE-11 are legacy/runtime baselines or rejected comparisons for new sales claims. Do not use them for new production training or promotion unless there is a future explicit exception with user approval, commercial-license review, local/runtime evidence, accuracy evidence, detector-off telemetry, and Jetson throughput evidence.

Web research gate:

- Before adding or replacing a model pack, run a fresh web/source check for that capability and prefer primary sources over blogs or benchmark roundups.
- Record each candidate's source URL, release/version date, license, model size, export formats, hardware/runtime notes, and the exact reason it was accepted, rejected, or kept as `demo_only`.
- Treat "latest" as a candidate filter, not a promotion rule. A newer model must beat the existing path on installability, license fit, YAML configurability, local evidence, Jetson/staging throughput, alert correctness, active-window telemetry, and false-positive guards.
- If the source is gated, paid, academic-only, non-commercial, or unclear, keep the claim blocked or pilot-only until we get explicit permission, a subscription approval, or a cleared replacement dataset.
- Refresh this source check whenever a model is promoted from `needs_work` or `pilot_ready` to `ready_to_sell`, and stale checks older than the current validation cycle are not enough for sales readiness.

Current source checks:

- OCR/ANPR: PaddleOCR 3.7.0 released PP-OCRv6 on 2026-06-11. Primary sources refreshed 2026-06-23: <https://github.com/PaddlePaddle/PaddleOCR>, <https://github.com/PaddlePaddle/PaddleOCR/releases>, <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>, <https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE>, and <https://arxiv.org/abs/2606.13108>. PaddleOCR upstream lists Apache-2.0 licensing, PP-OCRv6 tiny/small/medium tiers, and PP-OCRv6_medium as the current default general OCR pipeline model. The docs list PP-OCRv6_tiny_det as edge/IoT oriented, PP-OCRv6_small_det as the mobile balance point, and PP-OCRv6_tiny_rec as the smallest recognition tier. Use PP-OCRv6 tiny first for edge ANPR OCR latency, then PP-OCRv6 small if real gate footage shows too many misses. Keep PP-OCRv6 medium/server paths out of Jetson Nano-class real-time claims unless benchmarked. PP-OCRv6 is not a PPE object detector; do not use it for apron, harness, helmet, or other visible-PPE detection.
- ANPR public fixture: Wikimedia Commons source check refreshed 2026-06-22 for <https://commons.wikimedia.org/wiki/File:%27Vintage_Mg_car%27_at_%27Mumbai_Vintage_car_rally-2010%27.jpg>. The file page lists the self-published image as public domain / PD-self. Use it only as scoped public real-plate runtime evidence (`anpr_public_plate_image_read`), not as production gate footage, night/IR evidence, motion-blur evidence, or a commercial ANPR dataset.
- General detection/pose/segmentation: Ultralytics now lists YOLO26 as the current real-time model family. Primary sources refreshed 2026-06-23: <https://docs.ultralytics.com/models/yolo26/>, <https://docs.ultralytics.com/modes/export/>, <https://docs.ultralytics.com/guides/nvidia-jetson>, <https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE>, and <https://www.ultralytics.com/license>. YOLO26 has nano/small detection, segmentation, pose, classification, and oriented-box variants plus TensorRT/ONNX/CoreML/TFLite/OpenVINO export paths. Treat YOLO26n/s as candidates for new model packs, not automatic replacements, because Ultralytics documents AGPL-3.0/Enterprise licensing for YOLO26 code/models, and package compatibility, alert logic compatibility, and Jetson throughput still need proof.
- YOLO26 local runtime cleanup: `qa/video_eval/results/model_pack_device_probe.json` now gates the canonical model layout and checks for legacy YOLO11/YOLOE root, frontend, and old registry artifacts. `qa/video_eval/results/yolo26_false_positive_doctor.json` records the current local MPS negative-control run for `models/coco_primary/yolo26n.pt`; it passed with zero forbidden safety-class detections across blank, idle, office, construction, and warehouse fixtures at `conf=0.35`, `imgsz=640`. `qa/video_eval/results/yolo26_false_positive_doctor_conf010.json` is the lower-confidence stress companion at `conf=0.10`; it also passed with zero forbidden safety-class detections on the same fixtures. This is useful local false-positive evidence, but it does not replace Jetson three-camera throughput or customer-site false-positive testing.
- Open-vocabulary detection: Ultralytics documents YOLOE and YOLOE-26 as current open-vocabulary paths. Primary sources refreshed 2026-06-23: <https://docs.ultralytics.com/models/yoloe/> and <https://docs.ultralytics.com/models/yolo26/>. Keep YOLOE/YOLOE-26 as demo or benchmark candidates only until commercial licensing, prompt stability, Jetson throughput, and false-positive behavior are proven for each PPE class.
- Factory PPE/gloves: SH17 was checked 2026-06-20 as a public PPE source: <https://github.com/ahmadmughees/SH17dataset>. It includes 8,099 annotated images, 75,994 instances, 17 classes including gloves, glasses, masks, shoes, safety vest, helmet, medical suit, and safety suit, and published YOLOv8/v9/v10 benchmark weights. Use it for research/evaluation references only unless legal clears the dataset/license for our use; the repo lists CC BY-NC-SA 4.0 and educational/research-oriented responsible-use terms, so it is not an automatic commercial training base.
- Factory PPE/apron/harness: Primary checks updated 2026-06-22, with production model policy clarified on 2026-06-23. Open Images V7 official boxable classes do not include `apron` or `safety harness`: <https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv>. SH17 remains useful for several PPE classes but does not include apron or safety harness and is CC BY-NC-SA. SFCHD-SCALE covers safety helmet/clothing classes, not apron or safety harness, and the checked repository does not provide a clear commercial training license: <https://github.com/lijfrank/SFCHD-SCALE>. Roboflow Universe searches found apron seed sources and additional harness/work-at-height seed candidates, including Work at Height Safety, harness_detection, a Public Domain safety-harness source, a small full-body-harness source, and current Workspace FJWEPFJ1/eqjo apron sources with `00.person`, `01.apron`, and `02.no_apron` classes. These improve the seed queue and make apron person-box reconciliation reviewable, but they do not clear production because they still need export-term review, provenance review, class mapping, person-box approval, hard negatives, lanyard coverage, train/val/test split handling, approved seed-import manifests, and Jetson/runtime gates. The seed-source review queue now records `review_priority` and `review_focus`, with Work at Height Safety, harness-s4xxh, Workspace FJWEPFJ1/eqjo, Safety Food System, Kitchen Hygiene, and Kit ATT Det Apron Gloves as the first review targets. Current apron/harness evidence therefore stays local/pilot until we capture, procure, or approve cleared footage, annotate apron and harness/lanyard classes, train a closed-set YOLO26n/s candidate, and rerun local plus Jetson gates. The apron/harness production trainer rejects YOLO11/YOLOv8 fallbacks; those older families remain legacy/runtime baselines only unless a future explicit exception is added with evidence. Capture and annotation requirements are in `docs/plan/apron-harness-closed-set-ppe-dataset-plan.md`.
- Fire/smoke: Hugging Face model card checked 2026-06-20: <https://huggingface.co/odiug77/wildfire-smoke-fire>. It lists an Apache-2.0 YOLOv26m-based model for `smoke` and `fire` with 640px RGB input and T4 validation latency. Use it as the current local/staging candidate at the scoped `0.70` operating point, but do not claim certified fire-alarm performance, smoke-only lower-contrast events, broad indoor/hot-work false-positive robustness, or Jetson 3-camera throughput until those scenarios pass.
- Jetson staging: NVIDIA primary sources refreshed 2026-06-21: <https://developer.nvidia.com/embedded/jetson-nano-developer-kit>, <https://developer.nvidia.com/embedded/jetpack-archive>, and <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html>. The original Jetson Nano target has 472 GFLOPS, 4GB LPDDR4, 5-10W power, and NVIDIA's archive shows Nano support on JetPack 4.6.x while newer JetPack 6/7 entries target Orin/Thor-class devices. Treat original Nano, Orin Nano, Orin NX, and the current Jetson Orin staging box as separate benchmark targets. TensorRT deployment should use an export -> precision selection -> TensorRT engine -> runtime verification workflow. Treat any 3-camera claim as unproven until the exact target records FPS, latency, RAM, GPU load, and alert stability.
- Existing YOLO11/YOLOE paths remain legacy or pilot runtime evidence only. Do not choose them for new production model packs unless there is an explicit future exception with user approval, licensing review, runtime evidence, accuracy evidence, and Jetson throughput evidence.

Latest model/data sourcing notes from 2026-06-22 web search:

- "Latest" is a candidate filter, not the final selection. A model must still pass license, installability, export format, local MPS or CPU smoke test, Jetson benchmark, YAML configurability, alert behavior, detector-window telemetry, and false-positive evidence before it can replace a passing path.
- Current detector watchlist from 2026-06-22 source refresh:
  - Ultralytics YOLO26: primary Jetson/TensorRT candidate for future nano/small detector, pose, segmentation, and OBB packs, subject to AGPL/Enterprise licensing and side-by-side evidence against legacy runtime paths.
  - YOLO12: attention-centric real-time detector candidate; benchmark only if YOLO26 licensing/runtime or accuracy is not the right fit for a capability.
  - RF-DETR Nano/Small: Apache-2.0 real-time detector/segmentation candidate with 2026 open-source positioning; keep as a training/evaluation candidate until ONNX/TensorRT export, Jetson latency, and runtime integration are proven.
  - PP-OCRv6 tiny/small: current OCR/ANPR recognizer path; do not use for PPE/object detection because it is OCR, not a visual PPE detector.
- Web-researched model and dataset sources in `qa/video_eval/model_packs.yaml` now expire through `policy.source_research_max_age_days`; stale or malformed `checked` dates block the apron/harness seed-source audit until refreshed.
- YOLO26 is the current Ultralytics real-time family and is worth benchmarking for future nano/small packs. Its AGPL/Enterprise licensing and package/runtime compatibility still need proof before promotion, while older deployed YOLO11 baselines remain legacy evidence rather than new production candidates.
- PP-OCRv6 is the right ANPR recognizer candidate now because PaddleOCR 3.7.0 explicitly lists tiny/small/medium tiers, PP-OCRv6_medium as the default OCR pipeline model, and tiny/small tiers for edge/mobile tradeoffs. The next ANPR work should test PP-OCRv6 tiny on real or legally cleared gate footage first, then PP-OCRv6 small only if tiny misses too many plate reads.
- UFPR-VeSV was checked as a recent ALPR/vehicle dataset candidate: <https://github.com/Lima001/UFPR-VeSV-Dataset>. It is real-world surveillance data with plate annotations, but access requires a license agreement and is limited to academic non-commercial research, so it cannot be used as our commercial validation source without separate permission.
- LPLC was checked as a recent license-plate legibility dataset candidate: <https://github.com/lmlwojcik/lplc-dataset>. It is useful for understanding hard plate-quality cases, but access is by request and the repo states academic-only, non-commercial usage.
- SFCHD-SCALE was checked as an additional safety-clothing/helmet dataset candidate: <https://github.com/lijfrank/SFCHD-SCALE>. It has relevant safety-clothing/helmet annotations, but it does not cover the full RL-F PPE class set and the repository does not provide a clear commercial license gate. Treat it as research reference until licensing is cleared.
- Apron/harness public sourcing was checked again on 2026-06-23. Public seed sources now exist for apron and harness/work-at-height, but none are training-approved or production-complete. The refreshed source memo is `qa/video_eval/results/apron_harness_source_recheck_2026_06_23.md`, the same-day alternate-source scan is `qa/video_eval/results/apron_harness_alternate_source_scan_2026_06_23.md`, and the concise non-approving review kickoff is `qa/video_eval/results/apron_harness_source_review_kickoff.md`. Current top candidates are Work at Height Safety and harness-s4xxh for harness/lifeline coverage, Workspace FJWEPFJ1 and eqjo for apron/person/no_apron coverage, Safety Food System and Kitchen Hygiene for apron/no_apron volume, plus lower-priority supplemental sources already queued in `qa/video_eval/results/apron_harness_seed_source_review_checklist.csv`. The alternate Hugging Face/general-PPE sources found in the follow-up pass cover helmet/vest/shoes/mask/goggles/gloves-style PPE, not apron plus safety_harness/safety_lanyard, so they do not unblock the closed-set apron/harness production path. The correct next production step is priority-ranked approved source review, filled review evidence YAML, reviewed seed-import manifests, controlled capture/annotation, or an approved commercial dataset, not lowering YOLOE thresholds from the current local pilot configs. Roboflow-hosted model metadata and older YOLOv8/COCOn checkpoints are not accepted as new production models; the closed-set production path remains YOLO26n/s after reviewed data and side-by-side promotion gates.
- Apron/harness pretrained model search was refreshed on 2026-06-22. Public Hugging Face/GitHub PPE candidates found in the pass, including `Hexmon/vyra-yolo-ppe-detection`, `melihuzunoglu/ppe-detection`, `yihong1120/Construction-Hazard-Detection`, and similar YOLOv8/YOLOv11 safety projects, are not usable closed-set apron+harness promotion candidates as-is because they either cover helmet/vest/mask/glove-style classes without apron/harness/lanyard, or carry AGPL/Ultralytics derivative/commercial-license constraints that still require clearance and side-by-side runtime evidence. Keep production apron/harness on the capture/approved-dataset plus closed-set candidate path.
- If public/legal ANPR footage with readable plates cannot be found, the correct next step is not to fake confidence from synthetic footage. Keep production ANPR as `needs_dedicated_scenario` and ask for cleared customer gate footage, a paid/commercial dataset, or permission to record our own controlled gate clip.

Allowed:

- Open-source YOLO nano/small models.
- Open-source pose nano/small models.
- Lightweight open-source OCR only if deployable on edge hardware.
- Classical CV/tracking logic where it is more reliable than adding a model.
- Existing local model registry paths when they work.

Avoid for staging:

- Large VLMs in the real-time path.
- YOLOE as the production detector for Jetson Nano-class hardware.
- Heavy transformer detectors unless they are offline-only or benchmarked.
- Any model with unclear license for commercial use.

When no public model or dataset is good enough:

1. Use web search and browser automation to find public datasets, open model weights, papers, benchmarks, and license terms.
2. Prefer datasets with commercial-friendly terms.
3. If the best source is paid, gated, or subscription-only, stop and ask me before proceeding.
4. If customer footage is required, define the capture checklist and annotation spec instead of pretending public data proves it.

## Required Model Pack Shape

Add or validate model packs that can be installed and benchmarked:

- `base_3cam`: COCO nano model for person, vehicle, phone, animal, zone, queue, occupancy.
- `factory_ppe_3cam`: PPE specialist nano/small models for item-specific PPE.
- `fire_smoke_3cam`: fire/smoke specialist nano/small model.
- `pose_fall_3cam`: archived/backlog pose/person-down model contract only while RL-M is skipped.
- `anpr_gate_1cam`: plate detector + OCR, one gate stream only until benchmarked.

Current pack definitions live in `qa/video_eval/model_packs.yaml`. Treat that file as the staging handoff contract for model keys, source URLs, license notes, input sizes, recommended confidence, local validation commands, Jetson benchmark commands, unlocked claims, and claims that remain blocked. Local Mac device readiness is probed with `scripts/model_pack_doctor.py --out qa/video_eval/results/model_pack_device_probe.json`; if that result reports `mps_unavailable_cpu_fallback_only`, local CPU results prove software wiring only and do not satisfy the MPS performance gate. Saved YAML/runtime evidence is audited with `scripts/model_pack_evidence_doctor.py --out qa/video_eval/results/model_pack_evidence_doctor.json`; this checks model-pack scenarios, semantic YAML parsing, saved result status, YAML command evidence, screenshots, alerts, detector-window telemetry, and factory-PPE production-claim guardrails. Apron/harness pilot-vs-production status is audited with `scripts/apron_harness_readiness_doctor.py --out qa/video_eval/results/apron_harness_readiness_doctor.json`; this must keep production compliance blocked until the current model-pack evidence doctor passes for the same model-pack/result inputs, closed-set side-by-side promotion reports exist, and the `factory_ppe_3cam` Jetson full gate exists. The same apron/harness readiness report also records capture-manifest schema status, pilot and production label-count gaps, next capture batches, production capture matrix progress, dry-run training model, export format, and selected local training device. Open `qa/video_eval/results/apron_harness_capture_kickoff.md` first for the concise controlled-capture operator handoff, then use the full capture work order at `qa/video_eval/results/apron_harness_capture_work_order.md`; production training and candidate promotion must use `--capture-preflight-mode production` with `qa/video_eval/results/apron_harness_production_capture_matrix.csv` or an equivalent cleared production matrix. The generated label-review CSV now carries clip metadata columns; approved rows may create missing capture-manifest `clips` only when those reviewed metadata fields are filled, commercially cleared, and tied to non-repo raw storage. The closed-set model-registry slot is audited with `scripts/apron_harness_model_registry_doctor.py --planned-audit --out qa/video_eval/results/apron_harness_model_registry_report.json` before training, and must be rerun with `--candidate-report qa/video_eval/results/apron_harness_candidate_report.json --copy` only after the candidate and both side-by-side promotion gates pass for the same candidate-report SHA and selected-export SHA; the registry doctor enforces those promotion reports in copy mode. If the candidate used public/commercial seed imports, the registry doctor also requires the same `.seed_export_import.json` lineage sidecar that training, candidate promotion, side-by-side promotion, and readiness use, including preserved YOLO export preflight evidence for the exact ZIP SHA, zero orphan labels, and required local-class label counts.

2026-06-24 factory PPE production-gate handoff:

- Hospitals/RL-M remain skipped; the active handoff is factory PPE only.
- Current local runtime evidence keeps apron/harness `pilot_ready_not_production_compliance`, not production compliance.
- `qa/video_eval/results/apron_harness_model_registry_report.json` now records `registry_status=planned_no_candidate`: the `ppe_closed_set_candidate` model key is registered in code, but `models/ppe_closed_set_candidate/apron-harness-ppe.onnx` and its registry sidecar are intentionally absent until a candidate report is produced and copied.
- `qa/video_eval/results/apron_harness_dataset_pilot_schema_report.json` records the pilot schema gate: 0 approved labels today, with 300 missing each for `person`, `apron`, `safety_harness`, and `safety_lanyard` for 1,200 missing pilot annotations total.
- `qa/video_eval/results/apron_harness_dataset_production_schema_report.json` records the production schema gate: 0 approved labels today, with 1,000 missing each for `person`, `apron`, `safety_harness`, and `safety_lanyard` for 4,000 missing production annotations total.
- `qa/video_eval/results/apron_harness_seed_import_manifest_validation_summary.json` is the concise source-import gate handoff. It currently records `status=blocked_pending_import_manifest`, 25 candidate sources, 0 training-usable sources, 0 approved imports, and the next action to fill human/legal review evidence before any source can be used for training.
- `qa/video_eval/results/apron_harness_production_capture_matrix_validation_summary.json` is the concise production capture-matrix handoff. It currently records 21 rows, 0 ready rows, 2,404 missing labeled examples, 21 unapproved rows, and 21 unsafe/missing external storage references.
- `qa/video_eval/results/apron_harness_capture_kickoff.md` is the first page for a capture operator or SSH agent. Its immediate starter rows now include both positive and hard-negative rows for apron and harness, and it carries the starter validate/import/recheck command loop plus the required `.label_review_import.json` sidecar boundary before any training step.
- `qa/video_eval/results/apron_harness_candidate_runtime_runbook.md` is the runtime-only first page for the post-registration closed-set candidate. It lists the same six one-detection-at-a-time YAML scenarios, their backup/validate/plan/apply/run/restore commands, expected result JSON files, required evidence, and current blocked preflight status without the extra source-review, capture, registry, and Jetson context from the full promotion runbook. It also points to `scripts/apron_harness_candidate_runtime_runner.py`, whose plan mode lists the packet-derived sequence and whose execute mode refuses to run until `ppe_closed_set_candidate` is registered with a model artifact and registry sidecar.
- `qa/video_eval/results/apron_harness_production_gate_packet.json` is the machine-readable production-gate handoff for automation/agents. It carries the active factory-PPE scope, skipped hospitals/RL-M scope, gate status, grouped next actions, summary handoffs, candidate runtime contract, and guardrails that prevent premature model registration or production promotion. It now includes `candidate_runtime_execution_plan`, a six-step one-detection-at-a-time sequence for the post-registration closed-set candidate: apron active, apron false-positive guard, apron detector-window suppression, harness active, harness false-positive guard, and harness detector-window suppression. Each step pins the scenario ID, YAML config path, result path, `safetylens_site.py` backup/validate/plan/apply/restore commands, `scripts/video_eval.py run --scenario ...`, required evidence, current blocked preflight result, the standalone candidate runtime runbook path plus SHA-256, and the guarded candidate runtime runner path plus SHA-256. It also includes `candidate_training_execution_plan`, a reviewed-data-only sequence for training preflight, explicit `--run-training` candidate export, candidate doctor, side-by-side promotion reports, and registry copy after promotions pass; `scripts/apron_harness_candidate_training_runner.py` plan mode lists that sequence and execute mode refuses until reviewed production dataset YAML, capture manifest, seed-import manifest, and capture matrix exist, with actual training still requiring `--run-training`. The packet now also includes `jetson_gate_execution_plan`, a guarded four-step target-device path: stamp candidate identity into raw/soak templates, run raw Jetson benchmark, build the three-camera soak report, then execute `scripts/apron_harness_jetson_gate_runner.py --execute --json` to call `scripts/jetson_benchmark_doctor.py --require-full-gate`. That runner refuses until candidate report, raw benchmark, and soak report are supplied as existing non-placeholder paths, and the gate requires matching candidate-report SHA plus selected-export SHA across candidate, raw benchmark, soak, promotions, registry, and Jetson evidence. Its `first_unblock` section exposes both data paths: minimum public seed-source review artifacts plus a five-step `source_review_execution_plan`, and `controlled_capture_path`. The source-review plan validates the review bundle, fills the three minimum public-source evidence packets, validates the filled seed-import manifest, materializes only approved seed exports into a cleared manifest with a `.seed_export_import.json` sidecar, and reruns readiness while keeping production blocked unless production label counts, side-by-side promotion, registry, and Jetson gates pass. The same section now pins `scripts/apron_harness_source_review_runner.py`, whose plan mode lists the packet-derived source-review/import sequence and whose execute mode validates the review bundle but refuses to import seed exports until a filled non-placeholder seed import manifest passes `IMPORT_MANIFEST: gate=pass`, cleared capture/emit paths are supplied, and the emitted `.seed_export_import.json` sidecar validates after materialization. The controlled-capture path includes production capture matrix SHA, 21 required rows, 2,404 missing labeled examples, per-class deficits, required operator fields, full label-review validation/import commands, two sidecar-backed capture batches, four starter capture rows for denim apron positive, jacket hard-negative, fall-arrest harness positive, and backpack-straps hard-negative capture, plus the full and starter production label-review template paths, row counts, schemas, SHA-256 values, starter validate/import/recheck commands for an intermediate `apron_harness_capture_manifest.starter_reviewed.yaml`, starter success criteria covering `LABEL_REVIEW_VALIDATION: gate=pass`, the `.label_review_import.json` sidecar, starter-reviewed manifest validation, capture-matrix reconciliation, and the boundary that starter rows are not enough for production training/promotion. The same controlled-capture packet also pins `scripts/apron_harness_controlled_capture_runner.py`, whose starter plan mode lists the packet-derived review/import sequence and whose execute mode refuses label-review imports until a filled CSV and seed-import manifest are supplied, `LABEL_REVIEW_VALIDATION: gate=pass` succeeds, and the emitted `.label_review_import.json` sidecar validates after import. Command exit `0` alone is not enough for either data path. The same controlled-capture packet also pins the kickoff and work-order artifact paths plus SHA-256 values, validates that the kickoff still contains the starter validation loop while the work order still contains the full seed-review, label-import, seed-export-sidecar, and production training-preflight command chain, and carries a five-step `starter_execution_plan` for the next SSH agent: review rows, validate the starter CSV, import it, recheck the starter-reviewed manifest, and rerun readiness while keeping production blocked. Packet validation cross-checks the candidate runtime sequence against YAML template rows and runtime evidence, the candidate runtime runbook and runner paths plus SHA values, the candidate training runner path plus SHA and step order, the Jetson gate runner path plus SHA and full-gate step order, the source-review required sources and runner path plus SHA against the minimum review path, the controlled-capture runner path plus SHA, the starter execution-plan required rows against `starter_capture_rows`, and the validate/import/recheck commands against `starter_commands`. The packet also carries a four-item `post_capture_evidence_checklist` for the evidence that must exist after capture and before training: filled production matrix, filled production label-review CSV, reviewed manifest sidecar, and production training preflight.
- `scripts/apron_harness_readiness_doctor.py` writes both concise validation summaries and the production-gate packet by default during the final readiness recheck, and its generated promotion runbook includes explicit output paths, `jq` inspection commands for the full packet plus `candidate_runtime_execution_plan`, `candidate_training_execution_plan`, `jetson_gate_execution_plan`, `first_unblock.source_review_execution_plan`, `first_unblock.source_review_runner`, `first_unblock.controlled_capture_path`, `first_unblock.controlled_capture_path.runner`, starter rows, label-review templates, starter success criteria, starter execution plan, and post-capture evidence checklist, plus `--validate-production-gate-packet qa/video_eval/results/apron_harness_production_gate_packet.json --readiness-report qa/video_eval/results/apron_harness_readiness_doctor.json` so agents can prove the packet is fresh before using it. It also writes `qa/video_eval/results/apron_harness_candidate_runtime_runbook.md` by default and validates its SHA through the production gate packet. `backend/tests/test_apron_harness_seed_source_doctor.py` also guards validation-only source/import checks so they do not rewrite canonical handoff files or reviewer packet directories unless an output flag is explicitly passed.
- `qa/video_eval/results/factory_ppe_raw_benchmark.template.json` and `qa/video_eval/results/factory_ppe_3cam_soak.template.json` are the fillable Jetson target-device handoff templates. They must be filled from real target telemetry for the same candidate artifact SHA and candidate report SHA used by registry and promotion reports.
- Next executable sequence after approved data exists: validate/import reviewed labels, emit reviewed production capture manifest and dataset YAML, run `scripts/apron_harness_train.py --execute` with `--capture-preflight-mode production --require-capture-preflight`, run `scripts/apron_harness_candidate_doctor.py`, run apron and harness side-by-side promotion reports, rerun the registry doctor with `--candidate-report ... --apron-promotion-report ... --harness-promotion-report ... --copy`, then run `scripts/apron_harness_jetson_gate_runner.py --execute --json --candidate-report ... --raw-benchmark ... --soak-report ...`.
- The camera API and camera editor now round-trip `capability_model_overrides`, and the camera editor exposes a plain-language "Use trained apron/harness detector" control when apron or harness detections are selected. That control routes selected apron/harness capabilities to `ppe_closed_set_candidate` in preview and save requests, while the normal missing-model gate still blocks runtime until the candidate artifact and registry sidecar are present.

Each pack must define:

- model keys and files.
- download/source URL or local manual install path.
- license note.
- expected input size.
- recommended confidence.
- required runtime device.
- MPS local test command.
- Jetson staging benchmark command.
- feature claims it unlocks.
- claims it does not unlock.

ANPR OCR target:

- Use the existing plate detector plus PaddleOCR recognizer split.
- Preferred recognizer is PP-OCRv6 tiny first, then PP-OCRv6 small if tiny misses too many Indian plate reads.
- PP-OCRv6 tiny local evidence requires `paddleocr==3.7.0`, explicit PP-OCRv6 tiny detector/recognizer selection, model-server health capture, and fresh `scripts/video_eval.py` plate-read evidence. Do not claim PP-OCRv6 small, Jetson ANPR throughput, or production gate accuracy until those paths are benchmarked separately.
- Benchmark OCR latency separately from plate detection on Mac and Jetson; a one-gate ANPR camera is acceptable before any 3-camera ANPR claim.

## Runtime And Config Rules

Camera, rule, policy, and output setup must be YAML/config-driven.

Use:

```bash
python scripts/safetylens_site.py --config qa/video_eval/site.yaml validate
python scripts/safetylens_site.py --config qa/video_eval/site.yaml plan
python scripts/safetylens_site.py --config qa/video_eval/site.yaml apply --yes
```

For deployed staging:

```bash
python scripts/safetylens_site.py --config qa/video_eval/site.yaml apply --yes --restart
python scripts/safetylens_site.py doctor
```

Do not directly mutate `backend/config.json`, browser state, or DB rows to make a test pass. If YAML cannot express a required model, camera, rule, schedule, output, or threshold, extend the YAML loader first.

### Camera Event Policy YAML

Camera-local `event_policy` is the preferred deployment shorthand for "this camera sends these alerts this way." `backend/site_config.py` compiles it into a stable automation rule named `camera_event_<camera_id>` with preset `camera_event_default`; the stored camera config does not keep the shorthand field. Use this for alert/policy routing, not detector-off scheduling. Detector-off scheduling still belongs in `capability_windows`.

```yaml
alert_outputs:
  line_webhook:
    name: Line Webhook
    type: webhook
    enabled: true
    severities: [P1, P2]
    zones: []
    mode: live
    settings:
      url: ${LINE_WEBHOOK_URL}

cameras:
  line_1_apron:
    name: Line 1 Apron Camera
    zone: Packaging
    stream_type: rtsp
    rtsp_url: ${LINE_1_RTSP_URL}
    capabilities: [apron_required]
    safety_rule_ids: [ppe_apron]
    capability_windows:
      apron_required:
        mode: detection
        windows:
          - days: [mon, tue, wed, thu, fri]
            from: "08:00"
            to: "18:00"
    event_policy:
      output_ids: [line_webhook]
      severity: P2
      priority: 4
      cooldown_seconds: 90
      min_confidence: 0.67
      message_template: "{severity} {violation_type} on {camera} in {zone}"
      schedule:
        windows:
          - days: [mon, tue, wed, thu, fri]
            from: "08:00"
            to: "18:00"
```

Validation requirements: enabled policies need at least one `output_ids` entry; severity must be `P1` through `P4` or `inherit`; priority must be `1` or higher; cooldown must be non-negative; `min_confidence` must be between `0` and `1`; schedule windows need `from` and `to`.

### Deployed Camera Discovery YAML Flow

The SSH-facing discovery path can generate a reviewable site YAML file from ONVIF/RTSP discovery results. Real auto-discovery remains a deployment-test item until it is run on the target camera network, but the config-generation path is now scriptable.

Discovery-only review:

```bash
python scripts/safetylens_site.py discover 192.168.1.0/24 --timeout-seconds 5
```

Generate disabled cameras for review/editing:

```bash
python scripts/safetylens_site.py discover 192.168.1.0/24 \
  --output-site-yaml /etc/safetylens/discovered-cameras.yaml \
  --capabilities person_presence vehicle_presence \
  --event-output-ids in_app \
  --username-env SAFETYLENS_CAMERA_USERNAME \
  --password-env SAFETYLENS_CAMERA_PASSWORD
```

Discovery YAML includes an all-week `capability_windows` detector window and a per-camera `event_policy` by default. Use `--event-output-ids in_app floor_webhook` to set initial channels, or `--no-event-policy` when generating camera-only YAML for manual policy editing.

For factory PPE cameras after `ppe_closed_set_candidate` is registered, generate disabled apron/harness camera YAML with the closed-set candidate pinned at discovery time:

```bash
python scripts/safetylens_site.py discover 192.168.1.0/24 \
  --output-site-yaml /etc/safetylens/factory-ppe-cameras.yaml \
  --profile work_zone_ppe \
  --zone "Factory PPE" \
  --capabilities apron_required harness_required \
  --capability-model-override apron_required=ppe_closed_set_candidate \
  --capability-model-override harness_required=ppe_closed_set_candidate \
  --event-output-ids in_app browser_sound \
  --event-severity P2 \
  --event-priority 2 \
  --event-cooldown-seconds 45 \
  --event-min-confidence 0.20 \
  --username-env SAFETYLENS_CAMERA_USERNAME \
  --password-env SAFETYLENS_CAMERA_PASSWORD
```

After stream paths and credentials are verified, either edit `enabled: true` in the generated YAML or regenerate with `--enable-cameras`, then apply through the same YAML path. If the YAML uses credential env refs, set them before `apply`:

```bash
export SAFETYLENS_CAMERA_USERNAME=admin
export SAFETYLENS_CAMERA_PASSWORD='change-me'
python scripts/safetylens_site.py --config /etc/safetylens/discovered-cameras.yaml validate --allow-missing-env
python scripts/safetylens_site.py --config /etc/safetylens/discovered-cameras.yaml plan
python scripts/safetylens_site.py --config /etc/safetylens/discovered-cameras.yaml apply --yes --restart
python scripts/safetylens_site.py doctor
```

## One Detection At A Time Protocol

Every model/capability must be tested in isolation before it is tested in a bundled deployment.

For each detection:

1. Enable exactly one detection capability on exactly one test camera.
2. Disable unrelated safety rules, automation policies, and specialist model paths unless they are required dependencies.
3. Apply the scenario through YAML only.
4. Confirm the compiled execution plan loads only the expected model keys.
5. Run a positive clip where the detection should fire.
6. Run a negative or false-positive guard clip where it should not fire.
7. Run an active time-window test where detection and/or alerting is allowed.
8. Run an inactive time-window test where detection and/or alerting is blocked.
9. Record class counts, raw detections, alerts, policy decisions, delivery results, screenshots, and logs.
10. Restore the scenario to a clean state before moving to the next detection.

Only after the isolated test passes may the feature be tested in a bundled pack such as `factory_ppe_3cam` or `base_3cam`.

## Time Window Requirement

Every detection that can be sold must support day/week scheduling from config and UI.

Current baseline:

- The camera UI/policy work supports alert/policy scheduling: alerts, notifications, message templates, priority, severity, and cooldown can be scoped to time windows.
- This does not automatically mean the detector/model is disabled outside the time window.
- If a customer requirement says "the detection must not work outside these hours," that is detector-off scheduling and must be implemented separately.

There are two distinct scheduling modes:

| Mode | Meaning | Evidence required |
| --- | --- | --- |
| Detection window | The model/capability should run only during configured days and times. Outside the window, the camera may still stream, but this detection should not emit class telemetry or candidates. | In-window clip shows detections; out-of-window clip shows no detections/candidates for that capability. |
| Alert/policy window | The model may still detect outside the window, but alerts, notifications, messages, priority handling, and escalations should be suppressed. | In-window clip shows detection plus alert/delivery; out-of-window clip shows detections but zero matching alerts/deliveries. |

Default expectation:

- Use alert/policy windows for most safety use cases where historical detections remain useful.
- Use detection windows when a customer explicitly wants the detector disabled outside certain shifts, or when saving edge compute is required.

Implementation requirement:

- YAML must express daily and weekly windows for cameras, capabilities, or policies.
- UI must let an admin set active days, start time, end time, and whether the window controls detection or only alerting.
- Runtime telemetry must show whether a result was suppressed by schedule and which schedule applied.
- Detector-off telemetry must prove the capability did not run outside its active window, for example with skipped capability counters, model invocation counts, schedule IDs, and zero emitted detection candidates for that capability.
- `scripts/video_eval.py` must include at least two schedule assertions per detection: in-window positive and out-of-window suppression.

If the current runtime can only suppress alerts, mark full detection-window support as `needs_work` and implement capability-level active windows before claiming "detection is disabled outside shift hours."

## Detector-Off Scheduling Scope Of Changes

If this goal needs true detection-window behavior, the scope includes these changes before the claim can be marked sellable:

- YAML schema: add capability-level active windows per camera, with days of week, start time, end time, timezone behavior, and explicit mode such as `mode: detection` versus `mode: alert_policy`.
- Config loader: normalize and validate those windows through `scripts/safetylens_site.py validate/plan/apply`.
- Runtime planner: include active-window metadata in the camera execution plan so a disabled capability does not request its model path.
- Video processing loop: skip only the scheduled-off capability while keeping the camera stream alive and allowing other active capabilities to continue.
- Telemetry: record schedule suppression events with camera ID, capability key, schedule ID, current timestamp, suppression mode, zero emitted candidates, and model invocation counts so the test can prove detection was suppressed rather than only alerts.
- API/UI: show whether each capability is currently active, inactive by detection schedule, or alert-suppressed only; camera UI must keep event-channel/policy schedules distinct from detector active windows.
- API/UI: camera create/edit must preserve per-capability model overrides so apron/harness cameras can move from the generic PPE pilot path to the promoted closed-set candidate without losing event-policy channels or detector-window schedules.
- Video eval: assert both modes separately: no class telemetry for detector-off windows, and detections-with-zero-alerts for alert/policy windows.
- Failure rule: if an inactive detector window still emits detections/class telemetry and only suppresses notifications, the feature remains `needs_work` for any "detection itself is off outside shift" claim.
- Sales docs: use "alerts are active during configured windows" until detector-off scheduling evidence exists.

## Local M1 Execution Plan

1. Baseline current model registry.
2. Confirm MPS device selection and model readiness.
3. Run existing ready scenarios to avoid regressions.
4. Pick one detection gap: start with harness or apron.
5. Create an isolated one-detection YAML scenario.
6. Add active and inactive day/week windows for the detection.
7. Search for open model/dataset candidates if the existing model cannot detect it.
8. Download or train a lightweight candidate model if license allows.
9. Register the model in `backend/model_manager.py` only if it belongs in runtime.
10. Map it in `backend/capability_registry.py`.
11. Add or update scenario YAML and manifest entries.
12. Run `scripts/video_eval.py` in active-window mode and capture evidence.
13. Run `scripts/video_eval.py` in inactive-window mode and capture suppression evidence.
14. Update `SALES_READINESS_REPORT.md` and claim status.

## Staging Jetson Plan

1. Confirm exact hardware: original Jetson Nano, Orin Nano, Orin NX, or other.
2. Build a staging model pack with only the required features.
3. Export models to ONNX/TensorRT where practical.
4. Run 1-camera benchmark first.
5. Run 3-camera soak test with representative FPS.
6. Record:
   - FPS per camera.
   - inference latency.
   - CPU/GPU/RAM.
   - stream stability.
   - alert count and cooldown behavior.
   - model load time.
   - failure/restart behavior.
7. Do not claim 3-camera support until this benchmark passes.

## Test Gates

A feature is enabled only when these pass:

- Legal source/model license recorded.
- Public or commercial-dataset seed clips reference an approved seed-source review before training import.
- Public seed imports pass a filled import manifest review before clips are added to the capture manifest, including matching `source_review_sha256`, remote immutable `raw_export_ref`, `raw_export_sha256`, local reviewed YOLO export ZIP SHA-256, source-to-local class mapping, train/valid/test image and label presence, matching image files for every label file, and required local-class label-file counts; dataset/training preflight enforces `--seed-import-manifest` for any public or commercial seed clip.
- Local M1/MPS inference works.
- YAML can configure camera/rules/policies.
- Runtime camera reaches online/running.
- Stream renders.
- Detections appear with class telemetry.
- Alerts fire or suppress correctly.
- Active day/week window allows the detection or policy.
- Inactive `mode: detection` window skips the capability/model path and emits zero candidates for that capability; inactive `mode: alert_policy` window may still detect but must emit zero matching alerts/deliveries.
- False-positive guard clip exists where relevant.
- `scripts/video_eval.py` result JSON is saved.
- Screenshot or browser evidence exists.
- Sales readiness report is updated.

## Deliverables

- Updated feature/model matrix.
- Model pack definitions.
- New or updated model registry entries.
- New or updated capability mappings.
- Scenario entries in `qa/video_eval/manifest.yaml`.
- YAML config in `qa/video_eval/site.yaml`.
- Evidence JSON under `qa/video_eval/results/`.
- Active-window and inactive-window evidence for each sellable detection.
- Detector-off evidence showing `suppressedCapabilities`, schedule IDs, zero emitted candidates, and model invocation counts for inactive capability windows.
- UI scope for per-camera capability windows that separates detector scheduling from alert/policy scheduling.
- Updated `qa/video_eval/SALES_READINESS_REPORT.md`.
- Jetson staging benchmark report.
- Clear list of paid dataset/subscription asks, if any.

## Non Goals

- Do not rewrite the whole video pipeline before proving one model path at a time.
- Do not broaden marketing claims without evidence.
- Do not use paid datasets without approval.
- Do not sell clinical fall detection, identity matching, or fight detection unless dedicated validation exists.
- Do not target more than 3 cameras on Jetson Nano-class staging in this goal.
