# SafetyLens — UX Flows & Interaction Design

**Context for UX review:** This document breaks down every major user flow in the platform. Each flow specifies the actor, trigger, steps, decision points, error/edge states, and exit conditions. This accompanies `ui-ux-spec.md` which covers page layouts, design tokens, and architecture.

**Platform:** Factory safety video analytics dashboard (React SPA on Jetson edge device)
**Users:** Surveillance operators (8-12hr shifts), floor managers (periodic check-ins), admins (setup + config)
**Environment:** Control room monitors (often wall-mounted), office laptops, occasionally mobile browsers on factory floor

---

## User Personas

### Operator (Ravi)
- Surveillance room, monitors 10-20 cameras across 8hr shift
- Needs: immediate alert visibility, fast acknowledge, minimal clicks
- Tech comfort: basic — uses WhatsApp, can navigate simple web apps
- Pain: alert fatigue, too many clicks to acknowledge, missing critical alerts during bathroom break

### Floor Manager (Suresh)
- Walks the factory floor, checks dashboard 3-4 times per shift
- Needs: compliance overview, shift reports, escalation visibility
- Tech comfort: moderate — uses Excel, email, WhatsApp
- Pain: not knowing which zones are problematic, preparing audit reports manually

### Admin (Us — Techser)
- Sets up system at deployment, remote config changes
- Needs: camera setup, rule configuration, integration testing, troubleshooting
- Tech comfort: high
- Pain: configuring 20+ cameras with different rules efficiently, debugging false positives remotely

---

## Flow 1: Operator Shift — Monitoring & Alert Response

**Actor:** Operator (Ravi)
**Trigger:** Start of shift
**Frequency:** Continuous, 8-12 hours

### Happy Path

```
1. Open browser → SafetyLens loads at /live (default landing for operator role)
2. See camera grid (last-used layout preserved, e.g., 3x3)
3. Quick scan: all camera tiles show green status dots → all normal
4. [TIME PASSES — passive monitoring]

5. ALERT: P2 "No helmet" on Camera 7 (Welding Bay Entry)
   5a. Toast slides in (top-right): thumbnail + "No Helmet — Cam 7 Welding Bay" + "12s ago"
   5b. Camera 7 tile border flashes orange briefly
   5c. Right panel (alert feed) adds entry at top with orange severity bar

6. Operator clicks "Acknowledge" on toast
   6a. Toast dismisses
   6b. Alert status changes to "Acknowledged" in feed
   6c. Camera 7 tile border returns to normal
   6d. Backend logs: acknowledged by Ravi at 14:32:15

7. [ALTERNATIVE: Operator is away]
   7a. Toast persists for 15s, then auto-dismisses to drawer
   7b. Notification bell badge increments (+1)
   7c. After 10 min unacknowledged → escalation fires (Telegram to floor manager)
```

### P1 Critical Alert Path

```
1. ALERT: P1 "Person Fall Detected" on Camera 12 (Assembly Line)
   1a. Persistent toast (red, NO auto-dismiss) with alert sound
   1b. Camera 12 tile border pulses red continuously
   1c. Alert feed: red entry pinned to top
   1d. Telegram: photo sent to zone group immediately
   1e. PLC: buzzer activates in Assembly Line zone (holds until ack)

2. Operator clicks "Acknowledge" on toast
   2a. Toast dismisses, sound stops
   2b. PLC buzzer stops (signal sent to clear coil register)
   2c. Alert moves to "Acknowledged" status
   2d. Escalation timer cancelled

3. [ALTERNATIVE: Operator doesn't acknowledge within 3 min]
   3a. Escalation: SMS sent to floor manager (Suresh)
   3b. Alert feed shows "Escalated to Suresh at 14:35:00"
   3c. Toast remains on screen, sound continues

4. [ALTERNATIVE: False positive]
   4a. Operator clicks "View" on toast → alert detail sheet slides open
   4b. Sees camera snapshot with bounding box
   4c. Clicks "Mark False Positive" → adds to training feedback queue
   4d. Alert resolved with reason "False positive — marked by Ravi"
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Multiple P1 alerts simultaneously | All persist as toasts (stacked), each with sound. Most recent on top. Max 3 visible; overflow in drawer with badge count. |
| Same camera triggers same rule repeatedly | Dedup: within 60s window, update existing alert (increment count) rather than create new. Toast shows "No Helmet — Cam 7 (3rd occurrence)" |
| Camera goes offline during shift | P4 alert logged, no toast. Camera tile shows gray overlay + "Offline" badge. If camera was monitoring a hazardous zone, auto-escalate to P2 after 5 min. |
| Operator's browser tab is not focused | Browser notification API fires (if permitted). Sound still plays. Toast queues and shows when tab regains focus. |
| Operator logs out mid-shift | All unacknowledged alerts remain active. Next operator logging in sees them in the alert feed. No alerts are lost. |

---

## Flow 2: Floor Manager — Compliance Check

**Actor:** Floor Manager (Suresh)
**Trigger:** Periodic check (3-4 times per shift), or after receiving escalation
**Frequency:** 30 min sessions

### Happy Path

```
1. Open browser → lands on /dashboard (default for manager role)
2. See KPI row:
   - Incidents Today: 7 (↓ from 12 yesterday) — green trend
   - Compliance Rate: 94.2% — green
   - Active Cameras: 18/20 — amber (2 offline)
   - Avg Response Time: 2m 14s

3. Scan compliance heatmap:
   - Welding Bay / Night Shift = red cell (12 violations)
   - All other cells green/yellow

4. Click red cell → drills down to filtered alert list for Welding Bay + Night Shift
5. See pattern: most violations are "No helmet" between 02:00-04:00
6. Make note to brief night shift supervisor

7. Click "Reports" in sidebar
8. Select "Shift Report" → choose date range + zone
9. Click "Export PDF" → downloads report with:
   - Compliance % per zone
   - Top violations with camera snapshots
   - Response time distribution
   - Trend vs previous period
```

### Escalation Response Path

```
1. Suresh receives Telegram alert: "ESCALATED: Person Fall — Cam 12 Assembly Line — Unacknowledged 3 min"
2. Opens SafetyLens on phone browser
3. Sees persistent banner at top: "1 escalated alert requires attention"
4. Clicks banner → goes to Alert Center, filtered to escalated
5. Opens alert detail:
   - Sees snapshot with person on ground, bounding box
   - Sees timeline: detected 14:32 → toast shown → Telegram sent → unacknowledged 3 min → escalated
6. Calls floor to check on situation
7. Clicks "Acknowledge" + adds note: "Checked — worker tripped, first aid administered"
8. Alert resolved
```

---

## Flow 3: Admin — First-Time Camera Setup

**Actor:** Admin (Techser engineer)
**Trigger:** New site deployment
**Frequency:** Once per site, occasional additions

### Happy Path

```
1. Navigate to Configure > Cameras
2. Click "Add Camera" button (top-right)
3. Modal/sheet opens with two tabs: "Auto-Discover" | "Manual"

--- Auto-Discover Tab ---
4. Click "Scan Network"
   4a. Loading spinner: "Scanning network for ONVIF devices..."
   4b. After 5-15s: list of discovered cameras appears
   4c. Each row: IP address, model name, thumbnail preview (if available)
5. Select cameras to add (checkbox)
6. Click "Add Selected" (bulk action)
7. For each camera, a card appears in an "editing queue":
   - Auto-filled: IP, RTSP URL (from ONVIF profile)
   - Needs input: Display name, Zone assignment (dropdown)
8. Fill names + zones for each
9. Click "Test All" → each card shows green checkmark or red X with error
10. Click "Save All" → cameras added, appear in Live View grid

--- Manual Tab ---
4. Form fields:
   - Display Name: [text input]
   - RTSP URL: [text input with placeholder: rtsp://user:pass@192.168.1.100:554/stream1]
   - Zone: [dropdown, can create new zone inline]
5. Click "Test Connection"
   5a. Loading: "Connecting..."
   5b. Success: preview frame appears + "Connected successfully"
   5c. Fail: error message with troubleshooting hints
6. Click "Save"
```

### Error States

| Error | Message | Recovery |
|-------|---------|----------|
| No ONVIF devices found | "No cameras found on the network. Ensure cameras are powered on, connected to the same subnet, and have ONVIF enabled." | Manual add fallback, link to ONVIF setup guide |
| RTSP connection refused | "Could not connect to camera. Check URL format, credentials, and ensure camera is reachable from this device." | Inline edit to fix URL, re-test |
| Authentication failed | "Connection refused: invalid credentials. Check username and password." | Highlight credential fields in red |
| Stream timeout | "Camera reachable but stream timed out. Try a lower resolution stream profile." | Suggest alternative stream paths (e.g., /stream2, /sub) |
| Camera already exists | "A camera with this IP is already configured as 'Welding Bay - Entry'." | Link to existing camera, option to update |

---

## Flow 4: Admin — Detection Rule Configuration

**Actor:** Admin
**Trigger:** Initial setup or adding new detection capability
**Frequency:** Per deployment + occasional updates

### Simple Mode (Preset Scenarios)

```
1. Navigate to Configure > Detection Rules
2. See preset scenario cards in a grid:
   [PPE Detection] [Zone Intrusion] [Fall Detection]
   [Animal Detection] [Fire/Smoke] [Gangway Blockage]
   Each card shows: icon, name, description, status badge (enabled/disabled)

3. Click "PPE Detection" card
4. Expansion/modal shows:
   - Description: "Detects missing helmets, safety vests, and goggles"
   - Sub-rules (toggleable):
     [x] Hard hat detection
     [x] Safety vest detection
     [ ] Safety goggles (click to enable)
     [ ] Safety harness
   - Camera assignment: [Select cameras...] multi-select with zone grouping
   - Sensitivity: slider (Low / Medium / High) — maps to confidence threshold
5. Click "Enable"
6. Success toast: "PPE Detection enabled on 8 cameras"
```

### Advanced Mode (Custom Rule Builder)

```
1. Toggle "Advanced Mode" switch (top-right of rules page)
2. Click "+ New Rule"
3. Rule creation form (full-page or large sheet):

   --- Basic Info ---
   Rule Name: [text input]
   Description: [text area, optional]

   --- Detection Model ---
   Model: [YOLOE ▼] (dropdown: YOLOE / YOLO26 / YOLO-pose / VLM)

   --- Prompt Configuration (shown for YOLOE) ---
   Prompt Type: [Text ▼] (toggle: Text / Visual / Internal Vocab)

   [If Text]:
   Text Prompts: [tag input with autocomplete]
   - Type "hard hat" → autocomplete suggests from YOLOE vocabulary
   - Press Enter to add as tag
   - Multiple tags: "hard hat", "safety helmet"

   [If Visual]:
   Reference Image: [drag-and-drop upload zone]
   - Upload image → shows preview
   - Draw bounding box on uploaded image to isolate the reference object
   - Caption: "What object does this represent?" [text input]

   [If Internal Vocab]:
   Select Classes: [searchable checklist of 1200+ LVIS classes]

   --- Detection Logic ---
   Condition:
   [IF ▼] [Object Detected ▼] [hard hat ▼]
   [AND ▼] [NOT ▼] [Near Person ▼]
   [THEN ▼] [Create Alert ▼] [Severity: P2 ▼]

   (condition builder with add/remove row buttons)

   Confidence Threshold: [====|=====] 0.50
   Min Detection Duration: [0s ▼] (dropdown: 0s, 2s, 5s, 10s, 30s)

   --- Schedule ---
   Active: [Always ▼] (dropdown: Always / Time Range / Shift-based)
   [If Time Range]: Start: [08:00] End: [18:00] Days: [M T W T F S S]

   --- Camera Assignment ---
   [Select cameras...] grouped by zone
   "Select All in Zone" shortcut per zone group

4. Click "Preview / Test" button
   4a. Select a camera from dropdown
   4b. System grabs latest frame → runs rule → shows annotated result
   4c. Shows: bounding boxes, confidence scores, whether alert would fire
   4d. Admin can iterate: adjust prompt, threshold → re-test → see updated result

5. Click "Save Rule"
   5a. Rule appears in rule list, status: Active
   5b. Backend loads new YOLOE prompts without restart
```

### VLM Rule Configuration

```
1. In Advanced Mode, click "+ New Rule", select Model: "VLM (qwen3-vl)"
2. Additional fields appear:

   --- VLM Trigger ---
   Trigger Type: [Periodic ▼] (dropdown: YOLO-Gated / Periodic / Manual)

   [If Periodic]: Interval: [60 ▼] seconds (dropdown: 10s, 30s, 60s, 120s, 300s)
   [If YOLO-Gated]: Gate Rule: [select existing YOLO rule] — "fires when this YOLO rule triggers"

   --- VLM Prompt ---
   Question: [text area]
   Placeholder: "Describe what you want the AI to analyze..."
   Example: "Is the gangway between racks A and B clear and unobstructed? Are there any objects, materials, or equipment blocking the path?"

   --- Alert Condition ---
   Alert If Response: [Contains ▼] [text: "blocked, obstructed, not clear"]
   (dropdown: Contains / Does Not Contain / Severity Score Above)

   --- Expected Output ---
   Response Type: [Yes/No ▼] (dropdown: Yes/No, Description, Severity 1-5)

3. Preview: select camera → run VLM query → see:
   - Camera frame sent to VLM
   - VLM response text displayed
   - Alert evaluation result: "Would fire: YES — response contains 'blocked'"
```

---

## Flow 5: Admin — Alert Routing Configuration

**Actor:** Admin
**Trigger:** Initial setup or adjusting alert behavior
**Frequency:** Per deployment + periodic tuning

### Global Defaults Setup

```
1. Navigate to Configure > Alert Routing
2. Page shows three-tier tree: Global > Zones > Cameras (collapsible)
3. Click "Global Defaults" (root node)
4. Settings panel opens:

   --- Severity → Channel Matrix ---
   (table/grid with checkboxes)

              | In-App | Telegram | WhatsApp | SMS | PLC | Email Digest |
   P1 Critical|  [x]   |   [x]    |   [x]    | [x] | [x] |    [x]      |
   P2 High    |  [x]   |   [x]    |   [ ]    | [ ] | [ ] |    [x]      |
   P3 Medium  |  [x]   |   [ ]    |   [ ]    | [ ] | [ ] |    [x]      |
   P4 Low     |  [x]   |   [ ]    |   [ ]    | [ ] | [ ] |    [ ]      |

   --- Toast Behavior ---
   P1 Duration: [Persistent ▼] (locked for P1)
   P2 Duration: [15 seconds ▼] (dropdown: 5s, 10s, 15s, 30s, persistent)
   P1 Sound: [Alert Tone 1 ▼] [▶ Preview]
   P2 Sound: [Subtle Chime ▼] / [Off ▼]

   --- Timeout & Auto-Resolution ---
   (table, editable per rule category)

   | Rule Category      | Dedup Window | Throttle (max/hr) | Auto-resolve |
   |--------------------|-------------|-------------------|--------------|
   | Fire/Smoke         | 30s         | Unlimited         | Manual only  |
   | Person Fall        | 30s         | Unlimited         | Manual only  |
   | Zone Intrusion     | 60s         | 10/hr             | When cleared |
   | No Helmet          | 60s         | 10/hr             | 5 min        |
   | No Safety Vest     | 60s         | 10/hr             | 5 min        |
   | Gangway Blocked    | 120s        | 5/hr              | 10 min       |
   | Animal Detected    | 120s        | 5/hr              | 15 min       |
   | Camera Offline     | 300s        | 2/hr              | On reconnect |

   (each cell is editable inline)

   --- Escalation Chain ---
   Step 1: If unacknowledged after [3 ▼] min → Notify [Floor Manager ▼] via [Telegram ▼]
   Step 2: If still unacknowledged after [10 ▼] min → Notify [Plant Manager ▼] via [SMS ▼]
   Step 3: If still unacknowledged after [30 ▼] min → [Auto-log as missed ▼]
   [+ Add Step]

5. Click "Save"
```

### Zone Override

```
1. Expand "Battery Room" zone in the tree
2. Click "Battery Room" node
3. Settings panel shows inherited values (grayed out) with override toggles:

   "Inheriting from: Global Defaults" [Override ▼]

4. Toggle override for specific settings:
   - [x] Override severity: "All PPE violations in this zone → P1 Critical" (instead of default P2)
   - [x] Override channels: Add SMS for P2 alerts (not just P1)
   - [x] Override timeout: No helmet auto-resolve → 2 min (shorter, higher risk zone)

5. Overridden settings shown with orange "Override" badge
6. "Reset to Parent" button per setting to revert

7. "Apply to All Cameras in Zone" checkbox (checked by default)
   - Unchecked: only applies to new cameras added to this zone
```

### Per-Camera Override

```
1. Expand zone → click specific camera (e.g., "Cam 12 — Test Bay")
2. Same pattern: inherited from zone, with override toggles
3. Example override:
   - Disable "Animal Detection" rule (too many false positives from this camera angle)
   - Add PLC buzzer trigger for "Zone Intrusion" (this camera covers high-voltage area)

4. "Copy Settings From" dropdown:
   - Select another camera → imports its overrides
   - Useful when deploying multiple similar cameras
```

### Channel Configuration (sub-pages)

```
--- Telegram Setup ---
1. Click "Telegram" in channels sidebar
2. Form:
   Bot Token: [text input, masked] [Test ▼]

   Zone Groups:
   | Zone | Chat ID | Status |
   | Welding Bay | -100123456789 | Connected ✓ |
   | Battery Room | -100987654321 | Connected ✓ |
   | Assembly Line | [enter chat ID] | Not configured |
   [+ Add Zone Group]

   Message Template:
   [text area with variable chips]
   "⚠ {severity} ALERT: {rule_name}
   Camera: {camera_name} ({zone})
   Time: {timestamp}
   Confidence: {confidence}%"

   [Preview] button → shows rendered message
   [Send Test] button → sends test photo+message to selected group

--- PLC Setup ---
1. Click "PLC" in channels sidebar
2. PLC Devices:
   [+ Add PLC Device]

   PLC Name: [Assembly Line Panel ▼]
   IP Address: [192.168.1.50]
   Port: [502]
   Protocol: [Modbus TCP ▼] (dropdown: Modbus TCP / Siemens S7 / OPC-UA)
   [Test Connection] → "Connected — Modbus TCP, Unit ID 1"

   Register Mapping:
   | Alert Action | Register | Type | Signal | Duration |
   | Zone Buzzer ON | Coil 100 | Digital | Hold until ack | — |
   | Warning Light | Coil 101 | Digital | Pulse | 5 seconds |
   | Emergency Relay | Coil 102 | Digital | Hold until reset | — |
   [+ Add Mapping]

   [Test Signal] dropdown → select mapping → fires test → PLC responds

--- WhatsApp Setup ---
1. Click "WhatsApp" in channels sidebar
2. Status: "Requires WhatsApp Business API account"
3. Form:
   API Provider: [Meta Cloud API ▼] / [Twilio ▼] / [AiSensy ▼]
   API Key: [text input, masked]
   Phone Number ID: [text input]
   Template Name: [safety_alert_v1 ▼] (pre-approved templates)

   Recipients:
   | Role | Phone Number | Receives |
   | Plant Manager | +91 98765 43210 | P1 only |
   | Safety Officer | +91 87654 32109 | P1 + P2 |
   [+ Add Recipient]

   [Send Test Message] → sends template message with test image

   Cost Estimate: "~INR 0.13 per message. Estimated daily cost at current alert volume: INR 45"
```

### Testing Alerts Flow

```
1. Navigate to Configure > Alert Routing
2. Click "Test Alerts" button (top toolbar)
3. Test panel opens:

   Severity: [P1 Critical ▼]
   Camera: [Cam 7 - Welding Bay ▼]
   Rule: [No Helmet ▼]

   Channels to test:
   [x] In-App Toast
   [x] Telegram
   [ ] WhatsApp
   [ ] SMS
   [x] PLC Buzzer

   [Fire Test Alert]

4. Results:
   ✓ Toast displayed (persistent, with sound)
   ✓ Telegram sent to Welding Bay group (message ID: 12345)
   ✓ PLC Coil 100 activated (response: OK)

   [Dismiss All Test Alerts] → clears toast, stops buzzer, marks test alert as resolved

5. Test alerts are tagged as "[TEST]" in all channels and logs
   - Never count toward compliance metrics
   - Auto-resolve after 60 seconds if not manually dismissed
```

---

## Flow 6: Operator — Snooze & Suppress

**Actor:** Operator
**Trigger:** Known situation causing repeated alerts (maintenance, false positive pattern)

```
1. P2 toast appears: "No Helmet — Cam 5 Loading Dock"
2. Operator knows: maintenance crew is working without helmets (authorized)
3. Clicks "Snooze" on toast (or from alert feed)
4. Snooze options popover:
   - [15 min] [1 hour] [Until shift end] [Custom...]
   - Scope: [This camera only ▼] / [This rule on all cameras ▼] / [This zone ▼]
   - Reason (required for >1hr snooze): [text input]
5. Selects "Until shift end" + scope "Cam 5 only"
6. Confirmation: "No Helmet alerts snoozed on Cam 5 until 18:00"
7. Cam 5 tile shows small "Snoozed" badge (visible but not alarming)
8. Alert feed shows: "Snoozed by Ravi — No Helmet on Cam 5 until 18:00"
9. At shift end: snooze auto-expires, alerts resume
10. Manager can see all active snoozes in Alert Routing > "Active Snoozes" tab
```

---

## Flow 7: Manager — Shift Handover

**Actor:** Floor Manager (outgoing shift)
**Trigger:** End of shift
**Frequency:** Every shift change

```
1. Navigate to Reports > Shift Report (or click "Generate Shift Report" in Dashboard)
2. Auto-populated:
   - Shift: Day Shift (06:00 - 14:00)
   - Date: 2026-03-05
   - Zones: All (filterable)
3. Report preview shows:

   SHIFT SUMMARY
   ├── Total incidents: 14
   ├── P1 Critical: 1 (fall detection — resolved, first aid given)
   ├── P2 High: 8 (6 no-helmet, 2 zone intrusion)
   ├── P3 Medium: 5
   ├── Compliance rate: 91.3%
   ├── Avg response time: 1m 48s
   ├── Active snoozes: 1 (No Helmet on Cam 5, expires 18:00)
   ├── Cameras offline: 2 (Cam 3 since 11:20, Cam 14 since 13:05)
   └── Open/unresolved alerts: 3

   TOP INCIDENTS (with snapshots)
   1. [14:32] Person fall — Cam 12 Assembly Line — Resolved
   2. [09:15] Zone intrusion — Cam 8 High Voltage Bay — Resolved
   ...

4. Manager can add notes: "Cam 3 offline due to network issue, IT notified"
5. Click "Export PDF" or "Share via Telegram"
6. PDF saved + sent to shift group
7. Incoming manager opens report to review before taking over
```

---

## Flow 8: Admin — Remote Troubleshooting

**Actor:** Admin (remote, via VPN or cloud tunnel)
**Trigger:** Alert or call from site about false positives / missed detections

```
1. Navigate to System > Logs (or Alert Center filtered to specific camera)
2. Find problematic alerts → open detail
3. See: camera frame, bounding boxes, confidence scores, model used
4. If false positive pattern:
   a. Go to Configure > Detection Rules
   b. Find the rule → click "Edit"
   c. Adjust confidence threshold (e.g., 0.5 → 0.65)
   d. Click "Preview/Test" on problematic camera
   e. Verify: false positive no longer triggers, true positives still detected
   f. Save

5. If detection missed:
   a. Check camera frame quality (resolution, angle, lighting)
   b. Try adjusting YOLOE text prompt (more specific: "yellow hard hat" instead of "hard hat")
   c. If YOLOE insufficient: switch to VLM rule for this scenario
   d. If need more accuracy: flag frames for training data collection

6. Check System > Models page:
   - YOLO inference stats: FPS, avg latency, GPU utilization
   - VLM call stats: calls/hour, avg latency, cost
   - Model version info
   - Queue depth (if alerts are delayed, queue might be backing up)
```

---

## Flow 9: Always-Up Display (Wall Monitor)

**Actor:** System (unattended display)
**Trigger:** Configured by admin for control room wall mount

```
1. Admin navigates to Live View
2. Clicks "Always-Up Mode" toggle in toolbar (or accesses /live?mode=kiosk)
3. UI transforms:
   - Sidebar collapses completely (hidden)
   - Top bar minimized to thin strip: logo + clock + alert count
   - Camera grid fills full screen
   - Font sizes increase 1.5x
   - Camera labels: larger, higher contrast
   - Alert badges: larger, with pulsing animation for active alerts

4. Auto-cycle (if more cameras than grid slots):
   - Cycles through camera groups every 30s (configurable)
   - Pauses cycling when any camera has active P1 alert
   - Shows "Page 1/3" indicator at bottom

5. Alert overlay:
   - P1 alerts: full-width red banner across top with alert details
   - Banner persists until acknowledged (from any client — operator's workstation)
   - P2 alerts: camera tile border flashes, no banner

6. Browser configured in kiosk mode (F11 fullscreen)
7. Auto-recovery: if browser crashes or Jetson restarts, systemd service restarts browser to /live?mode=kiosk
```

---

## Flow 10: Bulk Camera Rule Assignment

**Actor:** Admin
**Trigger:** Need to apply same rules to many cameras efficiently
**Key concern from user:** "being able to set it up easily for all of them"

```
1. Navigate to Configure > Detection Rules
2. Select a rule (e.g., "PPE Detection")
3. Click "Manage Cameras" (or drag rule to zone in tree view)
4. Camera assignment modal:

   LEFT: Available Cameras (grouped by zone, with search)
   ├── Welding Bay
   │   ├── [ ] Cam 1 - Entry
   │   ├── [ ] Cam 2 - Overview
   │   └── [ ] Cam 3 - Workstation
   ├── Battery Room
   │   ├── [ ] Cam 4 - Charging Area
   │   └── [ ] Cam 5 - Assembly
   └── ...

   RIGHT: Assigned Cameras
   (shows currently assigned cameras)

   Quick actions:
   [Select All] [Select None] [Select Zone: ▼]

5. Admin clicks "Select Zone: Battery Room" → all 2 cameras checked
6. Admin clicks "Select Zone: Welding Bay" → all 3 cameras checked
7. Manually unchecks Cam 3 (not relevant angle)
8. Clicks "Apply" → rule now active on 4 cameras
9. Success: "PPE Detection enabled on 4 cameras (2 zones)"

--- Alternative: From Camera Config ---
1. Go to Configure > Cameras > select Cam 7
2. "Detection Rules" section shows all available rules with checkboxes
3. Toggle rules on/off for this camera
4. "Copy rules from..." dropdown → select another camera → applies same rule set
```

---

## Open UX Questions for Review

These are decisions we'd like a UX engineer to weigh in on:

1. **Alert feed position:** Right panel (always visible, eats screen space) vs. toggleable drawer (hidden until opened, might miss alerts)? Or both — panel on desktop, drawer on mobile?

2. **Camera grid interaction:** Click-to-expand (single camera takes over grid) vs. click-to-detail (opens side panel with camera detail)? Click-to-expand is more immersive but loses multi-camera context.

3. **Rule builder complexity:** Is the condition builder (`IF detected AND near person THEN alert`) too complex for the target users? Should we simplify to just "detect X → alert at severity Y" for v1?

4. **Zone drawing tool:** Canvas polygon tool on a floor plan vs. polygon tool on the camera feed itself (to mark ROI per-camera)? Or both?

5. **Mobile responsiveness:** How much do we invest in mobile layout? Operators use desktop monitors. Managers might check on phone. Is a responsive alert center + dashboard sufficient, or do we need a full mobile-optimized layout?

6. **Notification permission prompt:** When should we ask for browser notification permission? On first login? On first alert? On navigating to Live View?

7. **Onboarding:** Should there be a guided setup wizard for first-time use, or is the configuration flow documented above sufficient?

8. **Accessibility:** Factory control rooms can be noisy. Should we support visual-only alerting (flashing screen border, color changes) as an alternative to sound? Colorblind-safe palette for severity indicators?

9. **Multi-language:** TMEIC factory workers may prefer Hindi/regional language. Is i18n needed for v1, or English-only with plans for later?

10. **Alert sound customization:** One global alert sound, or per-severity/per-zone? Should operators be able to set their own preferences?
