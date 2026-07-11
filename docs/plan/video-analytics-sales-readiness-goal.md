# Video Analytics Sales Readiness Goal

## Paste Ready Goal Prompt

Run sales-readiness QA for Rakshak Lens video analytics across RL-E, RL-F, RL-H, and RL-O. Skip Hospitals/RL-M for the current validation cycle; fall, patient-monitoring, wandering, and clinical-care claims remain backlog until explicitly re-scoped. Use only YAML as the configuration mutation path: all cameras, streams, schedules, thresholds, zones, policies, notification channels, priorities, and messages must be represented in `qa/video_eval/site.yaml` and applied with `scripts/safetylens_site.py validate/plan/apply`. Do not directly edit active runtime config, database rows, localStorage, or browser-only state to make a test pass.

Test one deployment scenario at a time using public/legal video or a real RTSP camera when available. Before running, add the scenario to `qa/video_eval/manifest.yaml` with source, license/access note, expected detections, expected alerts, outputs, and sales claims. Apply YAML, run health checks, verify camera online state, stream rendering, detections, alert policy behavior, notification delivery, logs, and UI screenshots. Save structured evidence under `qa/video_eval/results`, update `SALES_READINESS_REPORT.md`, and update `CLAIMS_MATRIX.md`.

Separate alert/policy scheduling from detector-off scheduling. Alert/policy schedules may suppress notifications while detection still runs. If a claim says "detection itself should not run outside this shift," the scope must include capability-level active windows in YAML/runtime, UI controls that expose that mode, and telemetry proving the capability/model was skipped with zero emitted candidates outside the active window.

If a claim cannot be verified because telemetry is missing, implement the minimum telemetry needed first, then rerun. Classify every claim as `ready_to_sell`, `demo_only`, `needs_work`, `do_not_claim`, or `blocked`. Keep unsupported claims out of sales language until evidence exists.

## Objective

Prove what Rakshak Lens can safely sell across the current marketing deployments by testing the actual video analytics runtime, not just brochures or saved config.

A claim is sales ready only when the evidence chain is complete:

```text
video/RTSP source -> site YAML -> validate/plan/apply -> active runtime config -> camera online -> stream renders -> detections -> alerts/policies -> delivery results -> UI/API/log evidence -> report
```

Detection alone is not enough. For alerting claims, the configured threshold, cooldown, schedule, message, severity, priority, and output routing must all work at runtime.

Alert suppression is also not enough for detector-off claims. Those are sales ready only when capability-level active windows are applied through YAML, enforced before model/candidate emission, visible in the UI/API, and proven by telemetry.

## Non Negotiable Rule: YAML Only

All configuration changes for this goal must go through:

```bash
python scripts/safetylens_site.py --config qa/video_eval/site.yaml validate
python scripts/safetylens_site.py --config qa/video_eval/site.yaml plan
python scripts/safetylens_site.py --config qa/video_eval/site.yaml apply --yes
```

For deployed servers, use restart-aware apply:

```bash
python scripts/safetylens_site.py --config qa/video_eval/site.yaml apply --yes --restart
```

Do not manually patch `backend/config.json`, SQLite rows, browser state, or live API payloads to make a scenario pass. If YAML cannot express a required camera, zone, rule, output, schedule, or threshold, extend the YAML schema/loader/CLI first or mark the scenario blocked.

## Tomorrow SSH Runbook

Use this when logging into a deployed box with Claude/Codex.

1. Pull and inspect the deployed repo state.
2. Confirm services and URLs:

```bash
python scripts/safetylens_site.py doctor
```

3. Discover real cameras when on a customer/site network:

```bash
mkdir -p qa/video_eval/discovery
python scripts/safetylens_site.py discover --json > qa/video_eval/discovery/site-cameras.json
```

4. Convert discovered or known RTSP cameras into `qa/video_eval/site.yaml` entries. Each camera must declare its stream source, schedule, zones, rule bindings, notification outputs, priority, message template, and per-camera analytics settings.
5. Validate and preview changes:

```bash
python scripts/safetylens_site.py --config qa/video_eval/site.yaml validate
python scripts/safetylens_site.py --config qa/video_eval/site.yaml plan
```

6. Apply through YAML only:

```bash
python scripts/safetylens_site.py --config qa/video_eval/site.yaml apply --yes --restart
```

7. Run the relevant scenario and doctor checks:

```bash
python scripts/video_eval.py run --scenario <scenario_id>
python scripts/video_eval.py report
python scripts/safetylens_site.py doctor
```

8. If the run fails because evidence is invisible, add telemetry before lowering the claim or changing the product promise.

## Single Site YAML Contract

The goal is to make one site YAML capable of describing the whole deployment:

- Site name, timezone, config source, and merge behavior.
- Global runtime settings: FPS, model device, confidence, inference width, JPEG quality, cooldown defaults.
- Alert outputs: in-app, browser sound, Telegram, webhook, email, siren/PLC, or customer-specific channels.
- Safety rules and analytic capabilities.
- Automation policies with camera scope, conditions, schedules, severity, priority, message templates, cooldowns, and output IDs.
- Capability-level active windows for detector-off scheduling, distinct from alert/policy schedules.
- Cameras with `id`, name, zone, `stream_type`, `video` or `rtsp_url`, FPS, model mode, capabilities, safety rule IDs, zones, and per-camera analytic options.
- Zone polygons with normalized points, class filters, analytic type, and timing thresholds.

Every per-camera requirement must live in YAML, including:

- Notification channels.
- Alert schedule.
- Detection active windows for capabilities that should not run outside configured shifts.
- Priority.
- Message content.
- Detection windows.
- Zone geometry.
- Object classes or YOLOe prompts.
- Per-rule thresholds and cooldowns.
- Whether the camera is demo file-backed, RTSP, or discovered from site hardware.

## UI High Bar

The UI should become an editor for this same model, not a separate source of truth.

Minimum bar:

- Camera detail page shows active source, capabilities, zones, policies, schedules, notification outputs, recent detections, recent alerts, and health.
- Camera edit flow can modify the same fields represented in YAML.
- Per-camera notification routing is visible and testable.
- Zone editing is visual, with class filters and analytics type visible.
- Policy edits show severity, priority, schedule, message template, cooldown, and outputs.
- Camera capability scheduling distinguishes "run detection only during this window" from "detect always but alert only during this window."
- UI indicates whether config came from YAML and whether runtime has applied it.
- UI/API exposes whether each capability is currently active, inactive by schedule, or alert-suppressed only.
- Changes should be exportable back to site YAML or intentionally blocked as read-only when operating in YAML-controlled mode.

## Deployment Coverage

Map scenarios and claims to the current sales collateral:

| Code | Deployment | Required proof |
| --- | --- | --- |
| RL-E | Education | Phone usage, crowd/queue snapshot, after-hours/restricted zones, fire/smoke where claimed, RTSP/file deployment path. |
| RL-F | Factory | PPE, vehicle/person risk, route obstruction, object removal, fire/smoke, ANPR only where proven, per-camera notifications. |
| RL-H | Hospitality | Restricted entry, crowd/queue, route obstruction, object removal, fire/smoke, per-camera escalation. |
| RL-M | Hospitals/Medicare | Skipped this validation cycle; keep hospital/person-down scenarios as archived backlog evidence only. |
| RL-O | Office | After-hours intrusion, occupancy snapshot, crowd thresholds, vehicles/parking-adjacent scenarios, ANPR only where proven. |

## Video Source Rules

Use public/legal footage or customer-authorized RTSP streams. Record every source in `qa/video_eval/manifest.yaml`.

Each source entry needs:

- Source URL.
- License/access note.
- Local path or RTSP reference.
- Vertical and deployment type.
- Scenario ID.
- Expected visible event.
- Expected detections and alerts.
- Expected notification outputs.
- Whether the video can be committed, cached locally, or must remain external/private.

Large or restricted videos should not be committed unless the license allows it.

## Per Scenario Protocol

Run one scenario at a time.

1. Add or update the manifest scenario.
2. Define expected detections, alerts, notification outputs, and claim boundaries before running.
3. Add all camera/rule/output/zone/schedule config to `qa/video_eval/site.yaml`.
4. Validate, plan, and apply YAML.
5. Run doctor checks.
6. Confirm backend health and model readiness.
7. Confirm the camera exists and reaches online/running state.
8. Confirm stream rendering in UI/browser.
9. Let the clip or RTSP stream run long enough for inference cadence, thresholds, cooldowns, and schedules.
10. For schedule claims, run both modes intentionally: alert/policy suppression should show detections with zero alerts, while detector-off scheduling should show zero emitted candidates plus skipped-capability/model-invocation telemetry.
11. Capture detections, alerts, delivery results, schedule telemetry, screenshots, and logs.
12. Classify the scenario.
13. Fix one failure class at a time, then rerun the same scenario.

## Telemetry Fallback

If the system cannot prove a claim because evidence is missing, add minimum telemetry before judging the scenario.

Minimum useful telemetry:

- Structured `scripts/video_eval.py` result JSON.
- Camera lifecycle events: created, started, online, offline, error.
- Detection samples by camera, timestamp, class, confidence, zone, and source frame time.
- Alert creation records with rule, severity, policy ID, message, output IDs, and delivery results.
- Capability schedule telemetry: active/inactive decision, schedule ID, suppression mode, skipped capability count, model invocation count, and candidate count.
- Camera-specific analytic endpoints for non-basic analytics such as occupancy, queue, obstruction, object lifecycle, plates, or falls.
- Stream/UI screenshot capture.
- Backend log excerpts tied to scenario ID.
- Runtime health snapshot: model readiness, device, active cameras, config source, and recent errors.

Telemetry should make reruns reproducible from only the manifest, YAML, source video, and command log.

## Result Statuses

- `ready_to_sell`: End-to-end evidence passes on representative footage or real site camera.
- `demo_only`: Works on curated footage but is not robust enough for broad sales language.
- `needs_work`: Config, UI, detection, alerting, delivery, or telemetry fails in a fixable way.
- `do_not_claim`: Not built, not reliable, or not provable from current evidence.
- `blocked`: Missing source access, hardware, model, deployment access, or YAML/telemetry support.

## Current Evidence Baseline

The current QA report already has ready-to-sell evidence for:

- File-backed camera setup and simulated RTSP path.
- Person detection and camera-scoped alert routing.
- Phone usage alerting.
- PPE/missing helmet alerting on representative factory footage.
- Restricted-zone intrusion in hospitality and office footage.
- Crowd/person-count snapshot.
- Workstation occupancy snapshot.
- Public queue snapshot.
- Public vehicle presence.
- Public route obstruction snapshot.
- Watched-object removal lifecycle event.
- Observed watched-object dwell telemetry and camera-scoped dwell alert routing.
- Local HTTP webhook delivery with receiver-side payload evidence.
- Local SMTP email delivery with receiver-side message evidence.
- Local speaker/siren HTTP adapter delivery with receiver-side payload evidence.
- Local relay/buzzer HTTP adapter delivery with receiver-side payload evidence.
- Synthetic ANPR plate-read telemetry.
- Fire/smoke on public Wikimedia Commons footage, plus the older internal demo-pack fire/smoke clip.
- Historical fall/person-down and exercise/floor-therapy false-positive evidence remains archived backlog only while Hospitals/RL-M are skipped. Do not use it in current sales-readiness claims.
- Per-camera schedule suppression with visible detections and zero forbidden alerts during an inactive YAML operating window.
- Capability-level detector-off scheduling is now measured separately by `factory_apron_detector_window_suppression`: the camera stream stayed available, the apron-required capability was inactive by YAML active window, model invocation counts stayed at zero for the suppressed path, no detection candidates were emitted, and no forbidden alerts fired.
- Hairnet PPE is now measured through `factory_hairnet_false_positive_guard`: the runtime produced `person` plus hairnet-family class-count telemetry and zero `Missing hairnet` alerts after prompt expansion plus a camera-specific threshold override. Treat this as item-specific hairnet evidence with source-licensing caveat.
- Safety-vest PPE is now measured on public Pexels construction footage through `factory_vest_false_positive_guard`: the runtime produced `person` plus high-vis/reflective/construction-vest class-count telemetry and zero `Missing safety vest` alerts after a camera-specific threshold override. Treat this as item-specific vest evidence.
- Safety-goggles PPE is now measured on public Mixkit laboratory PPE footage: the runtime produced `person`, `safety goggles`, and `protective eyewear` class-count telemetry with zero `Missing safety goggles` alerts. Treat this as item-specific goggles evidence, not broad PPE compliance.
- Safety-boots PPE is now measured on public Mixkit hazmat/sanitation footage through `factory_boots_false_positive_guard`: the runtime produced `person`, `work boots`, `rubber boots`, and `steel-toe boots` class-count telemetry with zero `Missing safety boots` alerts. Treat this as item-specific boots evidence, not broad footwear certification.
- Face-mask PPE is now measured on public Mixkit laboratory footage through `factory_face_mask_false_positive_guard`: the rule-id/planner path runs COCO Primary plus PPE Specialist with expanded mask prompts and a camera-specific threshold override. The fresh runtime produced `person`, `procedure mask`, and `face covering` telemetry with zero `Missing face mask` alerts and fresh UI screenshot evidence. Treat this as item-specific face-mask evidence, not a respirator-fit or all-mask-condition guarantee.
- Face-shield PPE is now measured on a focused segment derived from public Pexels hospital PPE footage through `factory_face_shield_false_positive_guard`: the runtime runs COCO Primary plus PPE Specialist and produced `person`, `medical visor`, `ppe face shield`, and `shield mask` telemetry with zero `Missing face shield` alerts and fresh UI screenshot evidence. Treat this as item-specific face-shield evidence, not a guarantee across every shield/visor condition.
- Apron PPE is now measured on public Mixkit barista/food-service footage through `factory_apron_false_positive_guard`: the runtime produced `person` telemetry but no apron telemetry, and six `Missing apron` alerts fired on visible-apron footage. Treat apron compliance as `needs_work`, not ready.
- Gloves PPE is now measured on public Mixkit sanitation/PPE footage through `factory_gloves_false_positive_guard`: the runtime produced `person` plus gloves/blue-gloves class-count telemetry and zero `Missing gloves` alerts after a camera-specific threshold override. Treat this as item-specific gloves evidence.
- Safety-harness PPE is now measured on public Mixkit work-at-height footage through `factory_harness_false_positive_guard`: after adding first-class harness capability support, the runtime produced `person` telemetry but no harness telemetry, and six `Missing safety harness` alerts fired on visible-harness footage. Treat harness compliance as `needs_work`, not ready.
- Operational apply behavior: new cameras/policies should be applied with the backend stopped or through `scripts/safetylens_site.py apply --restart` on a deployed service, then verified with a clean plan and live API state.

Authoritative generated artifacts:

```text
qa/video_eval/SALES_READINESS_REPORT.md
qa/video_eval/CLAIMS_MATRIX.md
qa/video_eval/results/*.json
qa/video_eval/screenshots/
```

## Current Boundaries

Do not overclaim these until dedicated evidence exists:

- Real IP camera discovery and deployed-server auto-configuration on customer/site hardware.
- Full PPE-item compliance breadth beyond helmet. Missing helmet, safety goggles, safety boots, gloves, safety vest, hairnet, face mask, and face shield have item-specific passing evidence. Apron and harness still lack item telemetry and/or false-positive suppression.
- Certified accidental-fall accuracy or clinical patient-outcome claims. The current evidence supports scoped person-down/fall mechanics and floor-exercise suppression only.
- Calibrated crowd density, queue wait time/duration, and production-grade occupancy-duration analytics.
- Production ANPR accuracy on real customer gate footage.
- Tailgating, identity matching, face recognition, and badge/access correlation.
- Unattended-object threat assessment, owner association, abandoned-object intent, calibrated obstruction severity, and production parking-flow workflows.
- Aggression, fight-like movement, patient wandering, elopement, and clinical/patient-monitoring outcomes.
- Customer-environment notification outputs such as physical hooters/sirens, tower lamps, relay boards, PLCs, SMS, Telegram, customer SMTP/SendGrid email, or external SaaS webhooks unless tested in that deployment. The local speaker/siren and relay/buzzer HTTP adapter paths are proven; physical wiring and third-party acceptance are not.

## Required Artifacts

Maintain:

```text
qa/video_eval/manifest.yaml
qa/video_eval/site.yaml
qa/video_eval/results/*.json
qa/video_eval/screenshots/
qa/video_eval/discovery/
qa/video_eval/logs/
qa/video_eval/failures/
qa/video_eval/SALES_READINESS_REPORT.md
qa/video_eval/CLAIMS_MATRIX.md
test-videos/eval/
test-videos/public/
```

## Completion Criteria

The goal is complete when:

- Every brochure deployment has at least one representative runtime scenario, or the gap is explicitly classified.
- Real deployed-server config flow has been tested through `discover`, YAML edit, `validate`, `plan`, `apply --restart`, and `doctor`.
- Each tested scenario has structured result JSON, source provenance, screenshot evidence, and report coverage.
- `SALES_READINESS_REPORT.md` and `CLAIMS_MATRIX.md` clearly identify ready, demo-only, needs-work, blocked, and do-not-claim items.
- Per-camera notification behavior is proven for multiple cameras with different policies.
- Any missing observability has either been implemented or listed as a blocker.
- No scenario required bypassing the YAML configuration path.
