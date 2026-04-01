# SafetyLens UI/UX Specification

**Platform:** Video Analytics Safety Monitoring for Industrial Manufacturing
**Customer:** TMEIC (UPS Manufacturing Unit)
**Date:** 2026-03-05

---

## 1. Design Philosophy

**Guiding principle:** Show the data, hide the chrome. Every pixel serves a purpose.

This is not a hackathon dashboard or a government portal. It is a professional operations tool used 8-12 hours a day by surveillance managers and floor managers. The design draws from Linear (information density, collapsible sidebar, clean typography), Vercel/Geist (neutral color system, spacing tokens), and Grafana (alert routing, multi-tier configuration) — adapted for industrial safety context.

**Key constraints:**
- Light theme primary (factory control rooms are well-lit, dark themes cause glare on monitors)
- Must run on Jetson-served React app accessed via local network browsers
- Operators may have limited tech literacy — progressive disclosure over feature overload
- Wall-mounted monitors need a dedicated "always-up" mode with large text and high contrast

---

## 2. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | React + Vite (SPA) | Serves from nginx on Jetson, no SSR needed |
| Component library | shadcn/ui | Copy-paste model, Radix primitives, Tailwind, owns all source |
| Charts/analytics | Tremor | Purpose-built dashboard components, same Tailwind/Radix stack |
| Toast notifications | Sonner | 2-3KB, shadcn/ui native, custom JSX, stacking, programmatic dismiss |
| Data tables | TanStack Table (via shadcn DataTable) | Sorting, filtering, pagination for alert history |
| Styling | Tailwind CSS v4 | Utility-first, design tokens via CSS variables |
| Fonts | Geist Sans + Geist Mono | Clean, free, used by Vercel — professional without being corporate |
| Real-time | WebSocket (custom hooks) | Live detections, alert streaming from Python backend |
| State | Zustand | Lightweight, no boilerplate, good for real-time state |
| Icons | Lucide React | Consistent, clean, shadcn/ui default |

---

## 3. Design Tokens

### Typography

| Role | Font | Size | Weight | Use |
|------|------|------|--------|-----|
| Hero number | Geist Sans | 28px | 700 | KPI cards (incidents today, compliance %) |
| Page title | Geist Sans | 22px | 600 | Page headings |
| Section header | Geist Sans | 16px | 600 | Card titles, section labels |
| Body | Geist Sans | 14px | 400 | Tables, descriptions, form labels |
| Caption/meta | Geist Sans | 12px | 400 | Timestamps, camera IDs, secondary info |
| Code/technical | Geist Mono | 13px | 400 | RTSP URLs, model names, JSON configs |

**Line height:** 1.5 body, 1.25 headings. **Letter spacing:** -0.01em headings, 0 body.

### Color Palette (Light Theme)

**Neutrals (Geist-inspired):**

| Token | Hex | Use |
|-------|-----|-----|
| `bg-primary` | #FFFFFF | Main content background |
| `bg-secondary` | #FAFAFA | Sidebar, card backgrounds |
| `bg-tertiary` | #F5F5F5 | Hover states, nested cards |
| `border-default` | #E5E5E5 | Card borders, dividers |
| `border-active` | #D4D4D4 | Focused/active borders |
| `text-primary` | #171717 | Headings, primary content |
| `text-secondary` | #525252 | Body text, descriptions |
| `text-tertiary` | #A3A3A3 | Placeholders, disabled text |

**Semantic colors:**

| Token | Color | Hex | Background Hex | Use |
|-------|-------|-----|----------------|-----|
| `critical` | Red-600 | #DC2626 | #FEF2F2 | P1 alerts, fire, fall |
| `high` | Orange-500 | #F97316 | #FFF7ED | P2 alerts, missing helmet |
| `warning` | Amber-500 | #F59E0B | #FFFBEB | P3, PPE drift |
| `success` | Emerald-600 | #059669 | #ECFDF5 | Compliant, camera online |
| `info` | Blue-600 | #2563EB | #EFF6FF | Primary actions, links |

### Spacing

- Base unit: 4px
- Component padding: 12-16px
- Section gaps: 24-32px
- Sidebar: 240px expanded, 56px collapsed (icon-only)
- Border radius: 8px cards, 6px buttons, 4px inputs

---

## 4. Navigation & Layout

### Shell

```
+----------------------------------------------------------+
| [Logo] SafetyLens    [Zone Filter v]  [Search]  [Bell] [Avatar] |
+-------+--------------------------------------------------+
|       |                                                  |
| Side  |              Main Content Area                   |
| bar   |                                                  |
|       |                                                  |
|       |                                                  |
+-------+--------------------------------------------------+
```

### Sidebar Navigation (Collapsible, Linear-style)

**Monitoring**
- Live View — multi-camera grid with real-time detections
- Alert Center — all alerts with filtering, triage, acknowledge

**Analytics**
- Dashboard — KPI cards, compliance trends, heatmaps
- Reports — PDF/CSV export, shift reports, audit logs

**Configuration** (admin only)
- Cameras — add/remove/configure cameras
- Detection Rules — YOLOE text/visual prompts, VLM rules
- Zones — polygon ROI drawing, zone-camera mapping
- Alert Routing — channels, severity, timeouts, escalation
- Integrations — Telegram, WhatsApp, PLC, webhooks

**System** (admin only)
- Users — role management (admin, manager, operator)
- Models — YOLO/VLM model status, GPU usage, inference stats
- Logs — system logs, audit trail

### Top Bar
- **Zone filter dropdown** — filter entire view by factory zone (All / Welding Bay / Battery Room / Assembly Line / etc.)
- **Global search** — search cameras, alerts, rules by keyword
- **Notification bell** — badge with unread count, opens notification drawer
- **User avatar** — role indicator, logout, preferences

---

## 5. Page Specifications

### 5.1 Live View (`/live`)

**Purpose:** Real-time multi-camera monitoring with detection overlays.

**Layout:**
- Configurable grid: 1x1, 2x2, 3x3, 4x4 (toggle buttons in toolbar)
- Each camera tile shows:
  - Live video feed with YOLO bounding boxes overlaid (via canvas/WebGL)
  - Camera name + zone label (top-left overlay)
  - Alert status badge (top-right): green dot = normal, pulsing red = active alert
  - Detection count badge (bottom-right): "3 detections"
- Click tile to expand to single-camera detail view
- Drag-and-drop to rearrange camera positions in grid

**Right panel (toggleable, 320px):**
- Live alert feed — scrolling list of recent alerts with:
  - Severity color bar (left edge)
  - Camera thumbnail (48x48, cropped detection)
  - Alert type + camera name
  - Timestamp (relative: "12s ago")
  - Quick actions: Acknowledge / Snooze / Escalate

**Always-Up Mode** (for wall-mounted monitors):
- Toggle via toolbar button
- Hides sidebar and top bar
- Enlarges grid tiles, increases font sizes
- Auto-cycles through camera groups if more cameras than grid slots
- Full-screen capable (F11)

### 5.2 Alert Center (`/alerts`)

**Purpose:** Alert triage, investigation, and management.

**Layout:**
- **Filter bar** (top): severity, status (active/acknowledged/resolved/snoozed), camera, zone, rule type, date range
- **Alert table** (TanStack DataTable):
  - Columns: Severity (color dot), Timestamp, Camera, Zone, Rule, Confidence %, Status, Assigned To, Actions
  - Row click opens alert detail sheet (right slide-over panel)
  - Bulk actions: acknowledge selected, assign to, export
- **Alert detail sheet** (slide-over, 480px):
  - Full-size camera snapshot with detection bounding box highlighted
  - Alert metadata: rule name, confidence score, model used (YOLOE/VLM)
  - VLM reasoning (if applicable): "Why is this unsafe?" response
  - Timeline: created → notified (Telegram) → acknowledged (by whom) → resolved
  - Actions: Acknowledge, Escalate, Snooze (15m/1h/shift/custom), Mark False Positive, Add Note
  - Related alerts: other alerts from same camera/zone in last hour

### 5.3 Dashboard (`/dashboard`)

**Purpose:** Compliance overview and trend analysis.

**Layout:**
- **KPI row** (top, 4 cards via Tremor):
  - Incidents Today (number + sparkline vs yesterday)
  - Compliance Rate (% + trend arrow)
  - Active Cameras (count / total)
  - Avg Response Time (time to acknowledge)
- **Compliance heatmap** (center): zone x shift matrix, color-coded by violation count
- **Violation trend chart** (Tremor AreaChart): violations over time, filterable by rule type
- **Top violating zones** (Tremor BarList): ranked list of zones by violation count
- **Top violation types** (Tremor DonutChart): breakdown by rule (no helmet, no vest, zone intrusion, etc.)

### 5.4 Camera Configuration (`/configure/cameras`)

**Purpose:** Add, edit, and manage camera connections.

**Camera addition flow:**
1. **Auto-discover button** — scans network via ONVIF, lists found cameras with IP, model, preview thumbnail
2. **Manual add** — form: name, RTSP URL (`rtsp://user:pass@ip:554/path`), zone assignment
3. **Test connection** — pulls single frame, shows preview with "Connection successful" or error
4. **Save** — camera added to system, appears in Live View

**Camera list:**
- Card grid (not table) — each card shows:
  - Latest frame thumbnail
  - Camera name, zone, status (online/offline/error)
  - Active rules count badge
  - Quick actions: edit, disable, delete, view live

**Per-camera settings (edit view):**
- Connection: RTSP URL, credentials, resolution, FPS
- Zone assignment: dropdown (can belong to multiple zones)
- Detection rules: checkboxes to enable/disable rules for this camera
- Recording: toggle, retention period
- Alert overrides: per-camera severity/channel overrides (see section 6)

### 5.5 Detection Rules (`/configure/rules`)

**Purpose:** Configure YOLOE prompts, VLM rules, and detection logic.

**Two-tier interface:**

**Simple mode (default):**
- Preset scenario cards: "PPE Detection", "Zone Intrusion", "Fall Detection", "Animal Detection", "Fire/Smoke", "Gangway Blockage"
- Click card to enable → configure which cameras → done
- Each preset maps to pre-configured YOLOE/VLM prompts

**Advanced mode (toggle):**
- **Rule builder** — card-based, each rule has:
  - **Rule name** (e.g., "Helmet Detection - Welding Bay")
  - **Model selector**: YOLOE / YOLO26 / YOLO-pose / VLM
  - **Prompt type** (for YOLOE): Text / Visual / Internal Vocab
    - Text prompt field: with autocomplete from YOLOE's 1200+ built-in vocabulary
    - Visual prompt: image upload dropzone + crop tool to select reference region
  - **Detection logic**:
    - Condition builder: `IF [object detected/not detected] AND [in zone/near person] THEN [alert/log/PLC trigger]`
    - Confidence threshold slider (0.0-1.0, default 0.5)
    - Minimum detection duration (avoid transient false positives): 0s, 2s, 5s, 10s
  - **VLM rules** (separate section):
    - Trigger: YOLO-gated (fires when YOLO detects X) / Periodic (every N seconds) / Manual
    - Question prompt: free-text, e.g., "Is the gangway clear and unobstructed?"
    - Alert condition: "Alert if answer contains 'blocked' or 'obstructed'"
  - **Schedule**: always-on / time range / shift-based
  - **Assigned cameras**: multi-select with zone grouping

**Live preview panel:**
- Admin configures a rule → clicks "Test on Camera" → selects a camera → sees detections overlaid on a live frame
- Shows confidence scores, bounding boxes, and whether alert would fire
- Critical for validating YOLOE text prompts before going live

### 5.6 Zone Management (`/configure/zones`)

**Purpose:** Define physical zones and map cameras to them.

**Layout:**
- **Floor plan canvas** (center): upload factory floor plan image, draw polygon zones on it
  - Polygon drawing tool (click points to create zone boundary)
  - Color-code zones by risk level (red = hazardous, yellow = caution, green = general)
  - Place camera icons on floor plan to show coverage
- **Zone list** (right panel):
  - Zone name, risk level, assigned cameras count, active rules count
  - Click to edit zone properties

**Per-zone settings:**
- Name, description, risk level
- Assigned cameras (drag from available list)
- Active detection rules (inherit from global + zone overrides)
- Alert severity overrides (e.g., "No helmet in Battery Room" = P1 Critical instead of default P2)
- Alert channel overrides (e.g., Battery Room alerts also go to SMS)

---

## 6. Alert System Architecture

### 6.1 Severity Levels

| Level | Name | Color | Toast | Sound | Timeout | Example |
|-------|------|-------|-------|-------|---------|---------|
| P1 | Critical | Red #DC2626 | Persistent (no auto-dismiss) | Yes (configurable tone) | Manual acknowledge only | Fall, fire, zone intrusion into hazardous area |
| P2 | High | Orange #F97316 | 15 seconds | Optional | Auto-snooze after 10 min if unacknowledged | No helmet, forklift near-miss |
| P3 | Medium | Amber #F59E0B | No toast (drawer only) | No | Auto-resolve after 30 min if not re-triggered | PPE drift (goggles, gloves), gangway partial block |
| P4 | Low | Blue #2563EB | No toast (log only) | No | Auto-resolve after 60 min | Camera degraded, analytics confidence drop |

### 6.2 Toast Behavior (Sonner)

**Design:**
- Position: top-right corner
- Max visible: 3 toasts stacked (compact mode with depth effect)
- Each toast contains:
  - Severity color bar (left edge, 4px)
  - Camera thumbnail (40x40, detection crop)
  - Alert type (bold) + camera name
  - Timestamp
  - Action buttons: **Acknowledge** (primary) / **Snooze** (secondary) / **View** (ghost)

**Per-severity behavior:**

| Severity | Duration | Sound | Stacking | Auto-dismiss |
|----------|----------|-------|----------|--------------|
| P1 Critical | `Infinity` | Alert tone (configurable) | Always on top, pushes others down | Never — must click Acknowledge |
| P2 High | 15,000ms | Optional subtle chime | Normal stack | Yes, moves to drawer |
| P3 Medium | No toast | None | N/A | N/A |
| P4 Low | No toast | None | N/A | N/A |

**Programmatic control:**
- WebSocket delivers alert events → `toast.custom()` with custom component
- When another operator acknowledges (via WebSocket broadcast) → `toast.dismiss(id)` on all clients
- Testing mode: admin can trigger test toasts for any severity from Alert Routing config page

### 6.3 Alert Timeout & Auto-Resolution

**Scenario-based timeout defaults:**

| Scenario | Default Timeout | Auto-resolve Logic | Override Allowed |
|----------|----------------|-------------------|-----------------|
| Fire/smoke detected | No timeout | Manual only | No |
| Person fall | No timeout | Manual only | No |
| Zone intrusion (hazardous) | No timeout | Auto-resolve when person leaves zone (YOLO confirms) | Yes — can set to manual-only |
| No helmet | 5 min re-trigger window | If no re-detection in 5 min, auto-resolve | Yes — 1m to 30m |
| No safety vest | 5 min re-trigger window | Same as helmet | Yes |
| Gangway blocked | 10 min | If next VLM scan shows clear, auto-resolve | Yes — 5m to 60m |
| Animal detected | 15 min | If no re-detection in 15 min, auto-resolve | Yes |
| PPE drift (goggles, gloves) | 10 min | If next scan shows compliant, auto-resolve | Yes |
| Camera offline | 30 min | Auto-resolve when camera reconnects | Yes |

**Detection-linked auto-resolution:**
- For object-detection rules (helmet, vest, zone intrusion), the system checks subsequent frames
- If the violation condition is no longer true for N consecutive frames (configurable, default 30 frames / ~1 second at 30fps), the alert auto-resolves
- Resolution reason logged: "Auto-resolved: condition cleared" vs "Manually resolved by [user]"

### 6.4 Alert Fatigue Prevention

| Mechanism | How It Works | Default | Configurable |
|-----------|-------------|---------|-------------|
| **Deduplication** | Same camera + same rule + within time window = single alert (updated, not duplicated) | 60 second window | Yes — 10s to 5m |
| **Throttling** | Max N alerts per camera per rule per hour | 10/hour | Yes — 1 to unlimited |
| **Grouping** | Multiple cameras detecting same violation in same zone = single incident with sub-alerts | Enabled by zone | Yes — on/off |
| **Snooze** | Operator can snooze a rule on a specific camera for set duration | Presets: 15m, 1h, shift, custom | Yes |
| **Scheduled suppression** | Suppress non-critical alerts during maintenance windows or shift changes | Off by default | Yes — cron-style schedule |
| **Escalation chain** | If P1/P2 unacknowledged after N minutes, escalate to next tier | P1: 3min, P2: 10min | Yes |

### 6.5 Three-Tier Configuration (Inheritance Model)

```
Global Defaults
  |-- All cameras, all rules get these settings
  |-- Severity levels, timeout defaults, dedup windows
  |
  +-- Zone Overrides
  |     |-- Override severity, channels, timeouts for all cameras in this zone
  |     |-- Example: Battery Room → all PPE alerts upgraded to P1, add SMS channel
  |     |
  |     +-- Per-Camera Overrides
  |           |-- Override for a specific camera within the zone
  |           |-- Example: Camera 7 (entrance) → disable animal detection (too many false positives)
  |           |-- Example: Camera 12 (test bay) → add PLC buzzer trigger for zone intrusion
```

**UI for this:**
- Alert Routing page shows a tree view: Global → Zones → Cameras
- Click any node to edit its settings
- Overridden settings shown with an "override" badge and a "Reset to parent" option
- "Apply to all cameras" button at zone level for bulk configuration
- "Copy settings from" dropdown to clone configuration from another camera/zone

### 6.6 Notification Channels

| Channel | Library/API | P1 | P2 | P3 | P4 | Notes |
|---------|-------------|----|----|----|----|-------|
| In-app toast | Sonner | Yes (persistent) | Yes (15s) | No | No | Always on, not configurable off |
| In-app drawer | Custom component | Yes | Yes | Yes | Yes | Always on |
| Telegram | Bot API (`sendPhoto`) | Yes | Yes | Optional | No | Free, group per zone, image + caption |
| WhatsApp | Business API (template msg) | Yes | Optional | No | No | ~INR 0.13/msg, needs Meta approval, image via URL |
| SMS | Twilio / MSG91 | Yes | No | No | No | Escalation only, high cost |
| Email | SMTP / SendGrid | Digest | Digest | Digest | No | Shift report digest, not real-time |
| PLC/Buzzer | Modbus TCP (pymodbus) | Yes | Optional | No | No | Physical buzzer/light in zone |
| Webhook | HTTP POST | Yes | Yes | Yes | Optional | For custom integrations |

**Channel configuration UI:**
- Global defaults page: toggle channels on/off per severity level
- Per-zone/per-camera override: add or remove channels
- Each channel has its own config section:
  - **Telegram**: Bot token, chat IDs per zone, message template (supports variables: `{camera}`, `{zone}`, `{rule}`, `{timestamp}`, `{confidence}`)
  - **WhatsApp**: Business account, phone numbers, template ID
  - **PLC**: IP address, register address, signal type (momentary pulse / hold until ack), protocol (Modbus TCP / S7 / OPC-UA)
  - **Webhook**: URL, headers, payload template (JSON)

---

## 7. Workflow: First-Time Setup

For an admin deploying the system at a new factory:

### Step 1: System Setup
1. Deploy Jetson device, connect to factory network
2. Access `http://<jetson-ip>:3000` from any browser on the network
3. Create admin account (first-time setup wizard)

### Step 2: Add Cameras
1. Go to Configure > Cameras
2. Click "Auto-discover" — system scans network via ONVIF
3. Select discovered cameras, or manually add RTSP URLs
4. Test each connection (shows preview frame)
5. Name cameras meaningfully: "Welding Bay - Entry", "Battery Room - Overview"

### Step 3: Define Zones
1. Go to Configure > Zones
2. Upload factory floor plan image
3. Draw polygon zones on the floor plan
4. Assign risk levels (hazardous/caution/general)
5. Map cameras to zones

### Step 4: Configure Detection Rules
1. Go to Configure > Detection Rules
2. **Quick start (simple mode):** enable preset scenarios — PPE Detection, Zone Intrusion, Fall Detection
3. **Customize (advanced mode):** add custom YOLOE text/visual prompts for site-specific PPE
4. Test each rule on a live camera feed using the preview panel
5. Assign rules to cameras/zones

### Step 5: Set Up Alerts
1. Go to Configure > Alert Routing
2. Review global severity defaults (P1-P4)
3. Configure Telegram: enter bot token, create zone groups, send test message
4. Configure PLC (if applicable): enter PLC IP, register addresses, test signal
5. Set zone-specific overrides (e.g., Battery Room → upgrade all PPE to P1)
6. Configure escalation chains

### Step 6: Go Live
1. Go to Live View
2. Verify camera feeds are streaming
3. Verify detections are appearing (bounding boxes)
4. Trigger a test alert (admin tool) — verify toast, drawer, Telegram, PLC all fire
5. Hand off to surveillance operators

---

## 8. Role-Based Access

| Feature | Admin | Manager | Operator |
|---------|-------|---------|----------|
| Live View | Full | Full | Full |
| Alert Center (view/acknowledge) | Full | Full | Full |
| Alert Center (snooze/escalate) | Full | Full | Assigned zones only |
| Dashboard/Analytics | Full | Full | View only |
| Reports (export) | Full | Full | No |
| Camera Configuration | Full | No | No |
| Detection Rules | Full | View only | No |
| Zone Management | Full | View only | No |
| Alert Routing | Full | Zone-level overrides | No |
| User Management | Full | No | No |
| Model/System Config | Full | No | No |

---

## 9. Technical Architecture (Frontend)

```
React App (Vite, served via nginx on Jetson)
|
+-- WebSocket connection to Python backend (FastAPI)
|     +-- Real-time: detection events, alert events, camera status
|     +-- Bidirectional: acknowledge/snooze commands sent back
|
+-- REST API (FastAPI)
|     +-- CRUD: cameras, rules, zones, users, alert config
|     +-- Reports: compliance data, export endpoints
|
+-- State Management (Zustand)
|     +-- cameraStore: camera list, status, active feeds
|     +-- alertStore: active alerts, drawer state, unread count
|     +-- configStore: rules, zones, routing config
|     +-- authStore: current user, role, permissions
|
+-- Key Libraries
      +-- shadcn/ui: sidebar, cards, tables, forms, sheets, dialogs
      +-- Tremor: KPI cards, area/bar/donut charts, sparklines
      +-- Sonner: toast notifications
      +-- TanStack Table: alert history table
      +-- React Hook Form + Zod: form validation
      +-- Konva.js or Fabric.js: polygon zone drawing on floor plan/camera frame
```

---

## 10. Backend Integration Points

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `ws://host/ws/detections` | WebSocket | Stream real-time YOLO detections (bounding boxes per frame) |
| `ws://host/ws/alerts` | WebSocket | Stream alert events (new, acknowledged, resolved) |
| `GET /api/cameras` | REST | List cameras with status |
| `POST /api/cameras` | REST | Add camera (RTSP URL, name, zone) |
| `POST /api/cameras/{id}/test` | REST | Test camera connection, return preview frame |
| `GET /api/cameras/discover` | REST | ONVIF auto-discovery scan |
| `GET /api/rules` | REST | List detection rules |
| `POST /api/rules` | REST | Create detection rule (YOLOE/VLM config) |
| `POST /api/rules/{id}/test` | REST | Test rule on camera frame, return annotated image |
| `GET /api/alerts` | REST | List alerts with filters (severity, status, camera, zone, date) |
| `PATCH /api/alerts/{id}` | REST | Acknowledge, snooze, escalate, resolve |
| `GET /api/zones` | REST | List zones with camera mappings |
| `POST /api/zones` | REST | Create zone (polygon coords, risk level) |
| `GET /api/analytics/compliance` | REST | Compliance data for dashboard (KPIs, trends, heatmap) |
| `GET /api/reports/export` | REST | Generate PDF/CSV report |
| `POST /api/config/telegram/test` | REST | Send test message to Telegram |
| `POST /api/config/plc/test` | REST | Send test signal to PLC |

---

## 11. PLC Integration Details

### Supported Protocols

| Protocol | Library (Python) | Library (Node.js) | Common PLCs |
|----------|-----------------|-------------------|-------------|
| Modbus TCP | pymodbus | modbus-serial | Universal — nearly all PLCs |
| Siemens S7 | python-snap7 | NodeS7 | S7-1200, S7-1500, S7-300 (dominant in Indian factories) |
| OPC-UA | opcua-asyncio | node-opcua | Modern PLCs with OPC-UA server |

### PLC Actions

| Action | Trigger | Modbus Implementation | Duration |
|--------|---------|----------------------|----------|
| Zone buzzer ON | P1 zone intrusion | Write coil register (e.g., address 100, value 1) | Hold until alert acknowledged |
| Warning light | P2 alert | Write coil register (e.g., address 101, value 1) | Momentary pulse (5s) or hold until resolved |
| Emergency stop relay | P1 fire/explosion | Write coil register (e.g., address 102, value 1) | Hold until manual reset |
| All-clear signal | Alert resolved | Write coil register (e.g., address 100, value 0) | Instant |

### Configuration UI

Per-PLC device configuration:
- PLC name, IP address, port
- Protocol: Modbus TCP / S7 / OPC-UA
- Register mapping: alert type → register address → signal type (momentary/hold)
- Test button: sends test signal, confirms PLC response
- Heartbeat: periodic read to verify PLC is reachable (status indicator in System page)

---

## 12. Future Considerations

| Feature | Priority | Notes |
|---------|----------|-------|
| Mobile app (React Native) | Medium | Managers want alerts on phone; Telegram covers this for now |
| Floor plan live overlay | Medium | Show detection pins on floor plan in real-time |
| Face recognition (repeat violators) | Low | Privacy/legal review needed first |
| Multi-site support | Low | Single Jetson per site; cloud aggregation dashboard later |
| Shift handover report | Medium | Auto-generated at shift end with incidents, compliance %, camera status |
| Dark mode | Low | Add later via Tailwind CSS variable swap; light is primary |
| Offline mode | Medium | Queue alerts locally if network drops; sync when restored |
| Audit log | High | All config changes logged with user, timestamp, before/after values |

---

## 13. Sources & References

**Design Systems:**
- [Linear Design System (Figma)](https://www.figma.com/community/file/1222872653732371433/linear-design-system)
- [Vercel Geist Design System](https://vercel.com/geist)
- [Vercel Geist Font](https://vercel.com/font)

**Component Libraries:**
- [shadcn/ui](https://ui.shadcn.com/) — base components
- [shadcn/ui Dashboard Example](https://ui.shadcn.com/examples/dashboard)
- [Tremor](https://www.tremor.so/) — analytics components
- [Sonner](https://sonner.emilkowal.ski/) — toast notifications
- [TanStack Table](https://tanstack.com/table)

**Alert Patterns:**
- [PagerDuty Severity Levels](https://response.pagerduty.com/before/severity_levels/)
- [Grafana Notification Policies](https://grafana.com/docs/grafana/latest/alerting/configure-notifications/create-notification-policy/)
- [Carbon Design System - Notification Pattern](https://carbondesignsystem.com/patterns/notification-pattern/)
- [PatternFly - Notification Drawer](https://www.patternfly.org/components/notification-drawer/design-guidelines/)

**Surveillance Platforms (Competitive Reference):**
- [Vaidio Platform](https://www.vaidio.ai/platform)
- [Genetec Live Dashboards](https://resources.genetec.com/security-center-unified-security-platform/dashboards)
- [Milestone XProtect Rules Engine](https://doc.milestonesys.com/)
- [Verkada Camera Alerts](https://help.verkada.com/verkada-cameras/analytics/create-camera-event-alerts)
- [Frigate NVR](https://frigate.video/)

**Integration APIs:**
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [WhatsApp Business API](https://business.whatsapp.com/products/platform-pricing)
- [pymodbus](https://github.com/pymodbus-dev/pymodbus)
- [NodeS7](https://github.com/plcpeople/nodeS7)
- [ONVIF npm package](https://www.npmjs.com/package/onvif)

**Research:**
- [Real-Time Safety Monitoring Dashboards (Research Paper)](https://www.researchgate.net/publication/385214020)
- [Datadog Alert Fatigue Best Practices](https://www.datadoghq.com/blog/best-practices-to-prevent-alert-fatigue/)
- [incident.io Guide to Preventing Alert Fatigue](https://incident.io/blog/2025-guide-to-preventing-alert-fatigue-for-modern-on-call-teams/)
