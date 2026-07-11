# TMEIC Overnight Detection Hardening Goal

Updated: 2026-06-25

## Goal Prompt

Continue overnight in `/Users/gauthamgsabahit/workspace/techser/video-analytics`. Use the `safetylens-camera-setup` workflow and treat `docs/plan/tmeic-pilot-readiness-goal.md` plus `qa/video_eval/results/tmeic_pilot_readiness_matrix.json` as the source of truth. Work only through YAML/config for camera, rule, policy, active-window, and runtime changes. Prove behavior with `scripts/safetylens_site.py validate/plan/apply`, `scripts/video_eval.py`, telemetry, result JSON, screenshots, and focused tests. Do not claim production readiness without Jetson/site evidence.

Primary objective: finish only the TMEIC `can_do_now` pilot package. Do not spend this goal on `limited_pilot`, `needs_work`, or `blocked` requirements except to keep them explicitly out of scope in the docs/reports. For each `can_do_now` item, turn it on one at a time and validate active detections where applicable, false-positive guards, inactive capability windows, alert policy/channel behavior, and evidence reporting. Keep hospitals/RL-M skipped. Dog/cat animal detection can be piloted; snake remains unproven. Fire/smoke is scoped pilot-ready, not certified alarm/fire-panel ready.

The only in-scope TMEIC requirements for this goal are:
1. `TMEIC-01` Helmet alert needed in safety zones.
2. `TMEIC-03A` Dog/cat animal detection.
3. `TMEIC-04` Blockage in gangway.
4. `TMEIC-07` Head caps considered along with helmets in production/testing.
5. `TMEIC-10` Mobile usage while working.
6. `TMEIC-11C` Fire/smoke.
7. `TMEIC-13` Camera focused on test motor or drive.
8. `TMEIC-14` Risk zone around motor marked.
9. `TMEIC-15` Marked-zone entry triggers alert or buzzer.

Out of scope for this goal: `TMEIC-02`, `TMEIC-03B`, `TMEIC-05`, `TMEIC-06`, `TMEIC-08`, `TMEIC-09`, `TMEIC-11A`, `TMEIC-11B`, `TMEIC-11D`, `TMEIC-12`, `TMEIC-X1`, and `TMEIC-X2`. Leave their current statuses as-is unless a report needs wording to avoid accidental customer claims.

Start with the current `can_do_now` package:
1. Verify the matrix and reports present exactly 9 customer-pilot items as the deliverable package.
2. Re-run or validate the existing evidence for the 9 in-scope items only: positive/active where available, false-positive guard, detector-window suppression, and alert/channel behavior.
3. For every in-scope scenario, verify both schedule layers: alert/policy scheduling and capability-level detector active windows. Detector-off claims require zero detections, zero alerts, zero relevant model invocations/candidates, and telemetry proving the capability was inactive.
4. Update `docs/plan/tmeic-pilot-readiness-goal.md`, this doc, `qa/video_eval/results/tmeic_pilot_readiness_matrix.json`, `qa/video_eval/SALES_READINESS_REPORT.md`, and `qa/video_eval/CLAIMS_MATRIX.md` so the final customer-facing package is limited to `can_do_now`.

Use web/browser search only when needed for current public datasets or open-source models. Prefer 2025/2026 models that plausibly run on Jetson Nano for up to 3 cameras, then test locally on Mac M1/MPS first. If public data is insufficient, document the exact footage/subscription/site capture needed.

## Current Evidence Position

| Requirement | Current status | What we can say | What not to say yet |
| --- | --- | --- | --- |
| Fall/person-down | `limited_pilot` | Factory-only pilot evidence exists: active, false-positive, and detector-window TMEIC scenarios have passed. The pilot can alert on an already-fallen/person-down posture; it does not need to observe the falling motion itself. | Do not claim robust fall/person-down detection across crouching, bending, sitting, lifting, or maintenance-under-machine until those guards pass. |
| Fire/smoke | `can_do_now` for scoped pilot | Fire/smoke has public positive evidence, false-positive guards, detector-window suppression, and TMEIC PE Stores no-fire evidence. | Do not claim certified fire alarm replacement, smoke-only reliability, physical alarm integration, or Jetson throughput yet. |
| Dog/cat animal | `can_do_now` | COCO-backed dog/cat animal presence is pilotable with TMEIC no-animal guards and detector-window suppression. | Do not claim snake detection, species-level accuracy, night/IR performance, or small-animal reliability. |
| Snake | `needs_work` | `tmeic_snake_presence_main_gate_false_positive_probe` and `tmeic_snake_presence_main_gate_part2_false_positive_guard` prove a custom YOLOE long-tail snake prompt can run on two TMEIC no-snake main-gate clips with zero false detections/alerts. The paired detector-window scenarios prove the same capability can be suppressed outside YAML active windows with `yoloe_long_tail:0`. | Do not include in the customer pilot until positive snake footage, small-object sensitivity, night/IR validation, and production model quality are proven. |
| Plastic/snack cover in hand | `needs_work` | `tmeic_plastic_snack_cover_pe_stores_false_positive_probe` and `tmeic_plastic_snack_cover_main_gate_part2_false_positive_guard` prove the custom YOLOE long-tail prompt can run on PE Stores and outside-premises main-gate negative-control footage with zero false detections/alerts. The paired detector-window scenarios prove the same capability can be suppressed outside YAML active windows with `yoloe_long_tail:0`. | Do not claim positive recall, hand-object association, broader shopfloor robustness, or production readiness without positive clips and model validation. |
| Material fall | `needs_work` | `tmeic_material_fall_pe_stores_false_positive_probe` and `tmeic_material_fall_main_gate_part2_false_positive_probe` are useful failed probes: the prompt-only custom YOLOE path produced false `falling box` detections and Material Fall alerts on two normal TMEIC clips. The paired detector-window scenarios prove the same capability can be suppressed outside YAML active windows with `yoloe_long_tail:0`. | Do not imply person-fall/person-down covers falling materials, and do not claim material-fall detection until temporal trajectory logic, positive footage, and false-positive tuning are proven. |
| Drugs/medicines/syringes | `needs_work` | `tmeic_drugs_medicines_syringes_pe_stores_false_positive_probe` proves the custom YOLOE long-tail prompt can run on PE Stores negative-control footage with zero false detections/alerts. `tmeic_drugs_medicines_syringes_pe_stores_detector_window_suppression` proves the same capability can be suppressed outside a YAML active window with `yoloe_long_tail:0`. | Do not include in the customer pilot until policy/legal approval, acceptable-use boundaries, positive clips, small-object validation, and production model quality are proven. |
| Emergency boundary without PPE | `limited_pilot` | Co-configured zone/PPE relay scenario exists. | True compound correlation and physical buzzer/PLC wiring remain unproven. |

## Overnight Acceptance Criteria

- Every changed scenario has a YAML file, manifest entry, validation result, runtime result JSON, screenshot, and matrix/doc update.
- All 9 `can_do_now` items have a clear evidence path and customer-safe pilot wording.
- No `limited_pilot`, `needs_work`, or `blocked` item is included in the tomorrow deliverable package.
- Existing limited/unsupported work stays documented only as backlog: fall/person-down, harness/work-at-height, forklift/operator helmet association, air-gun/chemical goggles association, emergency boundary compound alerts, snake, drugs/medicines/syringes, material fall, face identification, explosion/abnormal events, and production Jetson/site scale.
- The final report states exactly what can be piloted tomorrow and avoids selling the remaining 12 requirements.
