# SafetyLens — UI Guide

**Version:** 2.0
**Date:** 2026-03-28

This document covers new and updated pages. The existing style guide (colors, typography, spacing, components) remains unchanged — refer to `safetylens-ui-styleguide.md` for design tokens.

---

## 1. Page Map

```
/live                          — Camera grid + single camera view
/alerts                        — Alert center (filter, acknowledge, resolve)
/dashboard                     — KPIs, charts, compliance
/search                        — AI Search (Phase 2)
/configure/cameras             — Camera CRUD + role assignment
/configure/rules               — Rules engine builder
/configure/zones               — Polygon zone drawing
/configure/alerts              — Alert routing matrix
/configure/plates              — ANPR plate lists (whitelist, blacklist)
/configure/faces               — Face enrollment + groups
/system/settings               — Global config, Telegram, VLM
/system/license                — License status + activation
/reports                       — Shift reports, export (Phase 3)
```

---

## 2. New Pages

### 2.1 /configure/rules — Rules Engine

This is the most important new page. It replaces the current detection rules page with a proper trigger → condition → action builder.

```
┌─────────────────────────────────────────────────────────────┐
│  Rules                                          + New Rule  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ● Gate Entry — Auto Open                     ON  ✎  ✕ │ │
│  │                                                        │ │
│  │   WHEN  plate_read  on  Gate 1 Entry                   │ │
│  │   IF    plate in Whitelist AND face in Employees       │ │
│  │   THEN  Open gate · Log entry                          │ │
│  │   ELSE  Alert P1 · Telegram · Hold gate                │ │
│  │                                                        │ │
│  │   Cooldown: 10s                          Last: 2m ago  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ● PPE — Helmet Required in Assembly                ON  │ │
│  │                                                        │ │
│  │   WHEN  detection "no_helmet"  on  Work Zone cams      │ │
│  │   IF    zone is Assembly Line AND confidence > 70%     │ │
│  │   THEN  Alert P2 · Telegram                            │ │
│  │                                                        │ │
│  │   Cooldown: 60s                         Last: 15m ago  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ● Fire — Emergency Response                        ON  │ │
│  │                                                        │ │
│  │   WHEN  detection "fire" OR "smoke"  on  All cameras   │ │
│  │   IF    confidence > 60%                               │ │
│  │   THEN  Alert P1 · Telegram · PLC siren · Webhook     │ │
│  │                                                        │ │
│  │   Cooldown: 30s                         Last: never    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Rule card anatomy:**
- Left: enabled dot (green = on, gray = off)
- Title: rule name, bold
- Body: human-readable summary of WHEN/IF/THEN/ELSE — not raw JSON
- Toggle: ON/OFF switch on right
- Edit (pencil) and delete (x) icons
- Footer: cooldown setting + "Last triggered" timestamp

#### Rule Editor (modal or slide-out panel)

```
┌─────────────────────────────────────────────────────────────┐
│  Edit Rule: Gate Entry — Auto Open                      ✕   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Name     [ Gate Entry — Auto Open                       ]  │
│                                                              │
│  ─── WHEN ──────────────────────────────────────────────── │
│  Trigger  [ plate_read         ▼ ]                          │
│  Cameras  [ Gate 1 Entry       ▼ ] [ + Add camera ]         │
│                                                              │
│  ─── IF ────────────────────────────────────────────────── │
│  Condition 1  [ plate_in_list  ▼ ]  List: [ Whitelist  ▼ ] │
│  AND                                                         │
│  Condition 2  [ face_in_group  ▼ ]  Group: [ Employees ▼ ] │
│  [ + Add condition ]                                         │
│                                                              │
│  ─── THEN ─────────────────────────────────────────────── │
│  Action 1  [ open_gate     ▼ ]  Device: [ boom_barrier_1 ] │
│  Action 2  [ log_entry     ▼ ]                              │
│  [ + Add action ]                                            │
│                                                              │
│  ─── ELSE (optional) ──────────────────────────────────── │
│  Action 1  [ create_alert  ▼ ]  Severity: [ P1 ▼ ]         │
│  Action 2  [ send_telegram ▼ ]                              │
│  Action 3  [ close_gate    ▼ ]  Device: [ boom_barrier_1 ] │
│  [ + Add action ]                                            │
│                                                              │
│  ─── Settings ─────────────────────────────────────────── │
│  Cooldown    [ 10  ] seconds                                 │
│  Priority    [ 100 ] (higher = evaluated first)              │
│                                                              │
│                                    [ Cancel ]  [ Save Rule ] │
└─────────────────────────────────────────────────────────────┘
```

**Key UX decisions:**
- Dropdowns for trigger, condition type, action type — not free text
- Each condition/action is a row with a type selector + type-specific fields
- AND logic between conditions (keep it simple — no OR/nested groups for v1)
- "ELSE" section is collapsible, hidden by default
- "Test Rule" button (dry-run against last 10 events, show what would have fired)

#### Preset Templates

When clicking "+ New Rule", offer presets to start from:

- PPE Violation (helmet/vest in work zone)
- Fire Emergency (fire/smoke → P1 + telegram + PLC)
- Gate Entry (ANPR + face → open/hold gate)
- After-Hours Intrusion (zone entry + time window)
- Overcrowding (person count > threshold)
- Custom (blank)

---

### 2.2 /configure/plates — ANPR Plate Management

```
┌─────────────────────────────────────────────────────────────┐
│  Plate Lists    [Whitelist] [Blacklist] [Visitors]  + Add   │
├─────────────────────────────────────────────────────────────┤
│  🔍 Search plates...                        Import CSV  ▲   │
│                                                              │
│  ┌──────────┬──────────────────┬───────────┬─────────────┐  │
│  │ Plate    │ Owner            │ Vehicle   │ Valid Until  │  │
│  ├──────────┼──────────────────┼───────────┼─────────────┤  │
│  │ KA05MN4523│ Rajesh Kumar    │ White Swift│ —           │  │
│  │ MH12AB1234│ Fleet Vehicle #3│ Blue Truck │ —           │  │
│  │ KA01HH9876│ Visitor: Suresh │ Gray i20   │ 2026-04-01  │  │
│  └──────────┴──────────────────┴───────────┴─────────────┘  │
│                                                              │
│  ─── Recent Reads ──────────────────────────────────────── │
│                                                              │
│  ┌──────────┬──────────┬──────────┬────────┬────────────┐  │
│  │ Plate    │ Camera   │ Time     │ Status │ Snapshot   │  │
│  ├──────────┼──────────┼──────────┼────────┼────────────┤  │
│  │ KA05MN4523│ Gate 1  │ 14:23:05 │ ✓ WL   │ [thumb]    │  │
│  │ TN09BC6543│ Gate 1  │ 14:21:18 │ ⚠ UNK  │ [thumb]    │  │
│  │ KA01HH9876│ Gate 2  │ 14:15:02 │ ✓ VIS  │ [thumb]    │  │
│  └──────────┴──────────┴──────────┴────────┴────────────┘  │
│                                                              │
│  Showing 156 reads today                      View all  ▶  │
└─────────────────────────────────────────────────────────────┘
```

**Layout:**
- Top half: plate list management (tab per list type)
- Bottom half: live read log (scrolling, auto-updates via WebSocket)
- Status badges: green "WL" (whitelist), red "BL" (blacklist), blue "VIS" (visitor), yellow "UNK" (unknown)
- Click a plate in the read log → option to add to a list
- Import CSV button for bulk plate upload
- Search works across all lists

---

### 2.3 /configure/faces — Face Enrollment

```
┌─────────────────────────────────────────────────────────────┐
│  Face Enrollment   [All] [Employees] [Visitors] [Contractors]│
│                                              + Enroll Face  │
├─────────────────────────────────────────────────────────────┤
│  🔍 Search by name...                                        │
│                                                              │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐            │
│  │ [photo]│  │ [photo]│  │ [photo]│  │ [photo]│            │
│  │        │  │        │  │        │  │        │            │
│  │ Rajesh │  │ Priya  │  │ Suresh │  │ Ahmed  │            │
│  │ Kumar  │  │ Sharma │  │ M.     │  │ Khan   │            │
│  │Employee│  │Employee│  │Visitor │  │Contract│            │
│  │ ✎  ✕  │  │ ✎  ✕  │  │ ✎  ✕  │  │ ✎  ✕  │            │
│  └────────┘  └────────┘  └────────┘  └────────┘            │
│                                                              │
│  ─── Recent Matches ────────────────────────────────────── │
│                                                              │
│  ● Rajesh Kumar matched at Gate 1         14:23:05  95.2%  │
│  ● Unknown face at Gate 1                 14:21:18  —      │
│  ● Priya Sharma matched at Gate 2         14:15:02  93.8%  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Enroll modal:**

```
┌─────────────────────────────────────────────────────────────┐
│  Enroll Face                                            ✕   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐                                        │
│  │                  │   Name    [ Rajesh Kumar            ]  │
│  │   [Photo area]   │   Group   [ Employees           ▼ ]  │
│  │   Drop image or  │   Valid   [ No expiry            ▼ ]  │
│  │   click to upload│                                        │
│  │                  │   OR capture from camera:              │
│  └─────────────────┘   Camera  [ Gate 1 Entry      ▼ ]     │
│                         [ Capture from live feed ]           │
│                                                              │
│  ☐  Person has given consent for facial recognition         │
│     (required under DPDPA)                                   │
│                                                              │
│                                    [ Cancel ]  [ Enroll ]   │
└─────────────────────────────────────────────────────────────┘
```

**Key UX decisions:**
- Grid of face cards (not a table — faces are visual)
- Consent checkbox is mandatory before enrollment
- Two enrollment methods: photo upload or live camera capture
- Group tabs filter the grid
- "Recent Matches" section at bottom shows real-time face match log
- Unknown faces highlighted with warning color

---

### 2.4 /search — AI Search (Phase 2)

```
┌─────────────────────────────────────────────────────────────┐
│  Search                                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 🔍  person with red helmet near forklift            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Cameras: [All ▼]   Time: [Today ▼]   Class: [All ▼]       │
│                                                              │
│  ─── 23 results ────────────────────────────────────────── │
│                                                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
│  │[crop]│ │[crop]│ │[crop]│ │[crop]│ │[crop]│ │[crop]│   │
│  │      │ │      │ │      │ │      │ │      │ │      │   │
│  │92.3% │ │89.1% │ │87.5% │ │85.2% │ │83.0% │ │81.4% │   │
│  │CAM-03│ │CAM-01│ │CAM-03│ │CAM-05│ │CAM-02│ │CAM-01│   │
│  │14:23 │ │14:21 │ │13:58 │ │13:45 │ │13:30 │ │13:12 │   │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘   │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                      │
│  │[crop]│ │[crop]│ │[crop]│ │[crop]│                      │
│  │      │ │      │ │      │ │      │                      │
│  │79.8% │ │78.3% │ │76.1% │ │74.5% │                      │
│  │CAM-07│ │CAM-03│ │CAM-01│ │CAM-05│                      │
│  │12:55 │ │12:40 │ │12:22 │ │12:10 │                      │
│  └──────┘ └──────┘ └──────┘ └──────┘                      │
│                                                              │
│                                              Load more  ▼  │
└─────────────────────────────────────────────────────────────┘
```

**Key UX decisions:**
- Large search bar at top — this is the primary interaction
- Results are a thumbnail grid (not a list) — visual search needs visual results
- Each thumbnail shows: cropped detection image, similarity %, camera, timestamp
- Click thumbnail → full frame with bounding box highlighted + context (what was detected before/after)
- Filters below search bar (camera, time range, detection class)
- Sorted by similarity score descending
- "Find similar" button on each thumbnail → re-searches using that image's embedding

---

### 2.5 /system/license — License Status

```
┌─────────────────────────────────────────────────────────────┐
│  License                                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Status:    ● Active                                        │
│  Customer:  TMEIC Jamshedpur                                │
│  License:   SL-2026-0001                                    │
│  Cameras:   7 / 10 used                                     │
│  Expires:   2027-03-26 (363 days remaining)                 │
│                                                              │
│  ─── Licensed Features ─────────────────────────────────── │
│                                                              │
│  ✓ Base Detections (10)                                     │
│  ✓ ANPR                                                     │
│  ✓ Face Recognition                                         │
│  ✗ AI Search (not licensed)                                 │
│                                                              │
│  ─── Activate New License ───────────────────────────────── │
│                                                              │
│  Drop .lic file here or click to browse                     │
│  [ Upload License File ]                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

- First-boot: this is the only page shown (full-screen activation overlay)
- After activation: accessible from System menu
- Expiry warning banner (yellow, 30 days before) shown on all pages
- Expired: red overlay on all pages, inference stops

---

## 3. Updated Pages

### 3.1 /configure/cameras — Camera Config (updated)

Add a **Role** dropdown to each camera card:

```
┌─────────────────────────────────────┐
│  CAM-01  Gate 1 Entry        ● LIVE │
│                                      │
│  RTSP:  rtsp://192.168.1.101/stream │
│  Zone:  Main Gate                    │
│  Role:  [ Gate (ANPR + Face)    ▼ ] │
│  FPS:   5                            │
│                                      │
│  Models: YOLO26n · YOLOE · ANPR     │
│          SCRFD · ArcFace             │
│                                      │
│  [ Edit ]  [ Delete ]               │
└─────────────────────────────────────┘
```

Camera roles:
- **General** — YOLO26n + YOLOE (default)
- **Gate (ANPR + Face)** — adds YOLO26s, PaddleOCR, SCRFD, ArcFace
- **Work Zone (PPE focus)** — YOLO26n + YOLOE with PPE classes prioritized
- **Custom** — manually select which models run

The role determines which models are loaded, which affects GPU load. Show estimated GPU impact when changing roles.

### 3.2 /live — Single Camera View (updated)

Add ANPR and face panels to the right sidebar for gate cameras:

```
┌──────────────────────────────────────┬──────────────────────┐
│                                      │  Gate Activity        │
│                                      │                      │
│         Gate 1 Entry                 │  14:23 KA05MN4523   │
│         (ANPR + Face overlays)       │  Rajesh K. ✓ WL     │
│                                      │  → Gate opened       │
│  ┌──────────────┐                    │                      │
│  │ KA05MN4523   │ ← plate overlay   │  14:21 TN09BC6543   │
│  └──────────────┘                    │  Unknown ⚠          │
│                                      │  → Alert P1 sent     │
│  ┌──────┐                            │                      │
│  │Rajesh│ ← face match overlay      ├──────────────────────┤
│  │95.2% │                            │  Recent Alerts       │
│  └──────┘                            │                      │
│                                      │  ● P1 Unknown 14:21 │
│                                      │  ● P4 Entry   14:23 │
│                                      │                      │
└──────────────────────────────────────┴──────────────────────┘
```

For gate cameras, the right panel shows:
- **Gate Activity** log — chronological feed of plate reads + face matches + gate actions
- Each entry: timestamp, plate, person name (or "Unknown"), list match status, action taken
- Color-coded: green (whitelist + known face), red (unknown/blacklist), yellow (partial match)

---

## 4. Navigation Update

```
┌─────────────────────────────────────────────────────────────┐
│  ┌──────┐  SafetyLens                                       │
│  │ Logo │  [Live] [Alerts] [Dashboard] [Search]             │
│  └──────┘                                                    │
│           Configure ▼          System ▼       3 Active ●    │
│           ├─ Cameras           ├─ Settings                   │
│           ├─ Rules             └─ License                    │
│           ├─ Zones                                           │
│           ├─ Alert Routing                                   │
│           ├─ Plates (ANPR)                                   │
│           └─ Faces                                           │
└─────────────────────────────────────────────────────────────┘
```

- **Search** added to top-level nav (Phase 2)
- **Configure** dropdown expanded with Plates and Faces
- **System** dropdown for Settings and License
- Feature-gated: Plates and Faces menu items hidden if not licensed

---

## 5. Gate Automation Flow (visual summary)

This is the end-to-end UX for the most complex use case — gate entry with ANPR + face recognition:

```
Vehicle approaches gate
        │
        ▼
Camera detects vehicle → ANPR reads plate
        │                       │
        │                 ┌─────┴─────┐
        │                 │           │
        │            In whitelist  Not in list
        │                 │           │
        ▼                 │           ▼
Face detected             │     Alert P1 → Telegram
        │                 │     Gate stays closed
   ┌────┴────┐            │     Guard investigates
   │         │            │
 Match    No match        │
   │         │            │
   │         ▼            │
   │    Alert P2          │
   │    "Known vehicle,   │
   │     unknown driver"  │
   │    Gate holds        │
   │                      │
   ▼                      │
Both match ◄──────────────┘
   │
   ▼
Gate opens automatically
Entry logged (plate + person + timestamp)
P4 info alert (for audit trail)
```

All of this is driven by the rules engine — no custom code per scenario. The admin configures rules through the UI, the engine evaluates them.
