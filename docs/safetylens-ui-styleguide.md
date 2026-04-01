# SafetyLens — UI/UX Style Guide

**Version:** 1.0
**Date:** 2026-03-23

---

## 1. Design Philosophy

SafetyLens is a **control room tool**. It will run on a big screen in a factory safety office, 24/7. Every design decision flows from this:

Inspired by **Linear.app** — adapted for industrial monitoring:

- **Speed is a feature** — every interaction < 50ms perceived latency. Optimistic UI everywhere.
- **Dark theme** — reduces eye strain on monitors running all day. Not pure black — dark blue-grays.
- **Quiet when nominal** — the UI is calm when everything is fine. Color and motion appear only for deviations. ("Dark cockpit" philosophy from aviation.)
- **High contrast** — critical alerts must be visible from 3 meters away.
- **Information density** — operators need to see many cameras at once. Tight padding, monospace for data.
- **Zero ambiguity** — severity must be instantly recognizable by color alone.
- **Keyboard-first** — operators shouldn't need a mouse for common actions. Cmd+K command palette.
- **Minimal clicks** — acknowledge an alert in 1 click, switch cameras in 1 click.

---

## 2. Color System

### 2.1 Base Palette (Dark Theme)

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg-primary` | `#0F1117` | Main background (near-black) |
| `--bg-secondary` | `#1A1D27` | Cards, panels, sidebars |
| `--bg-tertiary` | `#242836` | Hover states, active tabs |
| `--border` | `#2E3345` | Subtle borders, dividers |
| `--text-primary` | `#F0F2F5` | Main text (near-white) |
| `--text-secondary` | `#8B92A5` | Labels, metadata, timestamps |
| `--text-muted` | `#5A6178` | Disabled, placeholder text |

### 2.2 Severity Colors

These are **non-negotiable**. Every alert, badge, and indicator uses this system.

| Severity | Token | Hex | Background (10%) | Usage |
|----------|-------|-----|-------------------|-------|
| **P1 — Critical** | `--severity-critical` | `#FF3B3B` | `#FF3B3B1A` | Zone intrusion, fall, fire — immediate danger |
| **P2 — High** | `--severity-high` | `#FF8C00` | `#FF8C001A` | No helmet, no vest — safety violation |
| **P3 — Medium** | `--severity-medium` | `#FFD000` | `#FFD0001A` | Phone usage, minor violations |
| **P4 — Info** | `--severity-info` | `#3B82F6` | `#3B82F61A` | Headcount, general monitoring |
| **OK / Clear** | `--severity-ok` | `#22C55E` | `#22C55E1A` | Compliant, no violations |

### 2.3 Accent Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `--accent-primary` | `#6366F1` | Buttons, links, active elements (indigo) |
| `--accent-hover` | `#818CF8` | Hover state |
| `--accent-pressed` | `#4F46E5` | Pressed/active state |
| `--zone-outline` | `#38BDF8` | Zone polygon outlines on camera feed (sky blue) |
| `--detection-box` | `#A78BFA` | YOLO bounding boxes (purple, distinct from zones) |

---

## 3. Typography

### 3.1 Font Stack

| Usage | Font | Fallback | Why |
|-------|------|----------|-----|
| **UI text** | Inter | system-ui, -apple-system, sans-serif | Clean, highly legible at small sizes, excellent for data-heavy UIs |
| **Data / numbers** | JetBrains Mono | ui-monospace, monospace | Tabular numbers align perfectly, timestamps/counts are scannable |
| **Camera overlays** | JetBrains Mono | monospace | Must be readable over video at any resolution |

### 3.2 Scale

| Token | Size | Weight | Line Height | Usage |
|-------|------|--------|-------------|-------|
| `--text-xs` | 11px | 400 | 1.4 | Timestamps, metadata |
| `--text-sm` | 13px | 400 | 1.5 | Secondary labels, table cells |
| `--text-base` | 14px | 400 | 1.5 | Body text, descriptions |
| `--text-md` | 16px | 500 | 1.4 | Card titles, nav items |
| `--text-lg` | 20px | 600 | 1.3 | Section headings |
| `--text-xl` | 28px | 700 | 1.2 | Page titles |
| `--text-kpi` | 48px | 700 | 1.0 | KPI numbers on dashboard (JetBrains Mono) |

---

## 4. Layout

### 4.1 Page Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  Top Bar (48px)                                                  │
│  ┌──────┐  SafetyLens    [Live] [Alerts] [Dashboard] [Config]   │
│  │ Logo │                                    ┌────────────────┐ │
│  └──────┘                                    │ 3 Active Alerts│ │
│                                              └────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Content Area (fills remaining viewport)                         │
│                                                                  │
│  No sidebar. Full-width content.                                 │
│  Tabs for sub-navigation within each section.                    │
│                                                                  │
│                                                                  │
│                                                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**No sidebar.** Camera feeds need maximum horizontal space. Use top-level tabs for navigation.

### 4.2 Spacing Scale

| Token | Value | Usage |
|-------|-------|-------|
| `--space-1` | 4px | Inline element gaps |
| `--space-2` | 8px | Icon-to-text, badge padding |
| `--space-3` | 12px | Card inner padding (compact) |
| `--space-4` | 16px | Card inner padding (standard) |
| `--space-5` | 24px | Section gaps |
| `--space-6` | 32px | Page padding |
| `--space-8` | 48px | Major section dividers |

### 4.3 Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-sm` | 4px | Badges, small buttons |
| `--radius-md` | 8px | Cards, inputs, panels |
| `--radius-lg` | 12px | Modal dialogs, popovers |
| `--radius-full` | 9999px | Status dots, circular indicators |

---

## 5. Components

### 5.1 Camera Feed Card

```
┌─────────────────────────────────────┐
│ ┌─────────────────────────────────┐ │
│ │                                 │ │
│ │        Live Video Feed          │ │
│ │     (YOLO overlays drawn)       │ │
│ │                                 │ │
│ │  [person 0.94]  [hard_hat 0.87] │ │  ← bounding boxes in --detection-box
│ │                                 │ │
│ │  ┌─── RESTRICTED ───┐          │ │  ← zone polygon in --zone-outline
│ │  └──────────────────┘          │ │
│ │                                 │ │
│ └─────────────────────────────────┘ │
│  CAM-01 Production Floor    ● LIVE  │  ← green dot = connected
│  5 FPS · 2 violations today         │
└─────────────────────────────────────┘
```

- Aspect ratio: 16:9, never stretched
- Camera name: `--text-sm`, `--text-primary`
- Status dot: 8px circle, green=live, red=disconnected, yellow=degraded
- Stats: `--text-xs`, `--text-secondary`, JetBrains Mono for numbers
- On hover: subtle border glow (`--accent-primary` at 30% opacity)
- On click: expand to full-width single camera view

### 5.2 Alert Card

```
┌──────────────────────────────────────────────────────────┐
│ ●  P2 — No Helmet Detected                   2m ago  ✓  │
│    CAM-03 · Production Floor · Zone: Assembly Line       │
│ ┌──────────┐                                             │
│ │ snapshot │  Person detected without hard hat in        │
│ │  (thumb) │  restricted zone "Assembly Line".           │
│ └──────────┘                                             │
└──────────────────────────────────────────────────────────┘
```

- Left border: 3px solid, severity color
- Severity badge: pill shape, severity color background at 15%, text at 100%
- Timestamp: relative ("2m ago"), JetBrains Mono, `--text-secondary`
- Acknowledge button: checkmark icon, `--text-muted` → `--severity-ok` on click
- Thumbnail: 80x45px, rounded corners, from violation snapshot
- Unread alerts: slightly brighter background (`--bg-tertiary`)

### 5.3 KPI Card

```
┌────────────────────────┐
│  Violations Today      │
│                        │
│       23               │  ← --text-kpi, JetBrains Mono
│                        │
│  ▼ 12% vs yesterday   │  ← green if down, red if up
└────────────────────────┘
```

- Background: `--bg-secondary`
- Label: `--text-sm`, `--text-secondary`
- Number: `--text-kpi` (48px), `--text-primary`
- Trend: `--text-xs`, green (fewer violations = good) or red (more = bad)

### 5.4 Buttons

| Type | Style | Usage |
|------|-------|-------|
| Primary | `--accent-primary` bg, white text | Main actions (Save, Add Camera) |
| Secondary | `--bg-tertiary` bg, `--text-primary` text | Cancel, secondary actions |
| Danger | `--severity-critical` bg, white text | Delete, remove |
| Ghost | Transparent bg, `--text-secondary` text | Inline actions, icon buttons |
| Icon | 32x32, ghost style, rounded | Acknowledge, expand, settings |

All buttons: 36px height, `--radius-sm`, 500 weight, 13px text.

### 5.5 Tables

- Header row: `--bg-tertiary`, `--text-secondary`, `--text-xs`, uppercase
- Body rows: `--bg-secondary`, `--text-sm`
- Alternating rows: very subtle (`--bg-primary` / `--bg-secondary`)
- Row hover: `--bg-tertiary`
- Cell padding: `--space-3` vertical, `--space-4` horizontal
- Numbers: JetBrains Mono, right-aligned

### 5.6 Status Indicators

| State | Indicator | Color |
|-------|-----------|-------|
| Camera live | Pulsing dot (CSS animation) | `--severity-ok` (green) |
| Camera disconnected | Static dot | `--severity-critical` (red) |
| Camera degraded (<3 FPS) | Slow pulse dot | `--severity-medium` (yellow) |
| System healthy | Top bar badge | Green |
| System warning | Top bar badge | Yellow |
| License expiring | Banner at top of page | Yellow |
| License expired | Full-screen overlay | Red |

---

## 6. Page-Specific Layouts

### 6.1 /live — Camera Grid

```
┌─────────────────────────────────────────────────────────────┐
│  Live View          [2x2] [3x3] [4x4]    🔍 Search cameras │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  CAM-01  │ │  CAM-02  │ │  CAM-03  │ │  CAM-04  │       │
│  │          │ │          │ │          │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  CAM-05  │ │  CAM-06  │ │  CAM-07  │ │  CAM-08  │       │
│  │          │ │          │ │          │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

- Grid fills viewport height minus top bar
- Cameras fill cells proportionally (CSS grid, gap: 4px)
- Click camera → expands to single-camera view with alert sidebar
- Grid size selector persists in localStorage

### 6.2 /live?cam=X — Single Camera

```
┌─────────────────────────────────────────────────────────────┐
│  ← Back to Grid    CAM-03 Production Floor    ● LIVE  5fps │
├──────────────────────────────────────┬──────────────────────┤
│                                      │  Recent Alerts       │
│                                      │                      │
│         Large Camera Feed            │  ● P2 No helmet 2m  │
│         (with overlays)              │  ● P3 Phone    15m  │
│                                      │  ● P2 No vest  1h   │
│                                      │                      │
│                                      ├──────────────────────┤
│                                      │  Active Zones        │
│                                      │  ◼ Assembly Line     │
│                                      │  ◼ Restricted Area   │
│                                      │                      │
│                                      ├──────────────────────┤
│                                      │  Detection Modules   │
│                                      │  ✓ PPE              │
│                                      │  ✓ Phone            │
│                                      │  ✗ Fire (not lic.)  │
└──────────────────────────────────────┴──────────────────────┘
```

- Camera feed: 70% width
- Right panel: 30% width, scrollable
- Panel sections: collapsible

### 6.3 /alerts — Alert Center

```
┌─────────────────────────────────────────────────────────────┐
│  Alerts    [All] [P1] [P2] [P3] [P4]    Today ▼  CAM ▼    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ● P1 Zone Intrusion — CAM-05              30s ago  ✓  │ │
│  │   Person entered restricted zone "Motor Bay"          │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ● P2 No Helmet — CAM-03                    2m ago  ✓  │ │
│  │   Person in Assembly Line zone without hard hat       │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ● P3 Phone Detected — CAM-01              15m ago  ✓  │ │
│  │   Cell phone usage in production area                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Showing 47 alerts                              Load more ▼ │
└─────────────────────────────────────────────────────────────┘
```

- Filter pills at top: severity, time range, camera
- Newest first, infinite scroll or "Load more"
- Click alert → expands to show snapshot + full details
- Bulk acknowledge button for selected alerts

### 6.4 /dashboard — KPI Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  Dashboard                                    Shift: Day ▼  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Violations│ │Compliance│ │ Cameras  │ │ Avg Resp │       │
│  │   23     │ │  87%     │ │  8/10    │ │  45s     │       │
│  │ ▼12%     │ │ ▲ 3%    │ │ 2 down   │ │ ▼ 8s     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
│  ┌──────────────────────┐ ┌────────────────────────────┐    │
│  │  Violation Trend     │ │  Compliance Heatmap        │    │
│  │  (24h line chart)    │ │  (zone x shift matrix)     │    │
│  │                      │ │                            │    │
│  │  ╱╲    ╱╲            │ │  Zone A  ■■■□□            │    │
│  │ ╱  ╲╱╱  ╲           │ │  Zone B  ■■■■□            │    │
│  │╱        ╲╱          │ │  Zone C  ■■□□□            │    │
│  └──────────────────────┘ └────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Top Violations by Camera                             │   │
│  │  CAM-03 ████████████ 12                               │   │
│  │  CAM-01 ████████ 8                                    │   │
│  │  CAM-05 █████ 5                                       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

- KPI cards: 4 across, CSS grid
- Charts: Recharts (already in project) or lightweight alternative
- Heatmap: custom component, color = compliance % (green → red)
- All data refreshes every 30s via WebSocket

---

## 7. Camera Overlay Design

The overlays drawn on the camera feed (bounding boxes, zones, labels) must be visible over any background.

### 7.1 Bounding Boxes

```
┌─ person 0.94 ──────────────┐
│                             │
│                             │
│                             │
└─────────────────────────────┘
```

- Stroke: 2px solid
- Color by class:
  - `person`: `#A78BFA` (purple)
  - `hard_hat`: `#22C55E` (green)
  - `safety_vest`: `#22C55E` (green)
  - `no_helmet` (violation): `#FF3B3B` (red)
  - `cell_phone`: `#FFD000` (yellow)
  - `dog` / `cat`: `#FF8C00` (orange)
  - `fire` / `smoke`: `#FF3B3B` (red, pulsing)
- Label: class name + confidence, `--text-xs`, monospace
- Label background: class color at 80% opacity, rounded top corners
- For violations: box pulses (CSS animation on overlay canvas)

### 7.2 Zone Overlays

- Zone polygon: 2px dashed stroke, `--zone-outline` (`#38BDF8`)
- Zone fill: `--zone-outline` at 8% opacity
- Zone label: zone name, centered in polygon, `--text-xs`, white text with dark shadow
- Active violation in zone: polygon fill changes to severity color at 15% opacity

### 7.3 Overlay Rendering

- Use HTML5 Canvas overlaid on MJPEG `<img>` element
- Canvas redraws at frame rate (5-15 FPS)
- Bounding box coordinates received via WebSocket (not baked into MJPEG stream — allows toggling overlays without re-encoding video)

---

## 8. Animations & Transitions

Keep animations **functional, not decorative**.

| Element | Animation | Duration | Purpose |
|---------|-----------|----------|---------|
| New alert toast | Slide in from right | 300ms | Draw attention |
| Alert card appear | Fade in + slight slide up | 200ms | Smooth list update |
| Camera status change | Dot color transition | 500ms | Smooth state change |
| P1 alert | Pulse border (repeating) | 1.5s | Demand immediate attention |
| Fire/smoke detection box | Pulse opacity (repeating) | 1s | Indicate active danger |
| Page transitions | None (instant) | 0ms | Speed > aesthetics |
| Modal open | Fade in + scale from 95% | 150ms | Context |
| Loading states | Skeleton shimmer | Continuous | Indicate loading |

---

## 9. Responsive Behavior

SafetyLens is primarily a **desktop/large-screen** application. But it should degrade gracefully.

| Breakpoint | Layout | Usage |
|------------|--------|-------|
| >= 1920px | Full layout, 4x4 camera grid | Control room monitor |
| 1280-1919px | Full layout, 3x3 camera grid | Desktop |
| 768-1279px | Stacked layout, 2x2 grid | Tablet / laptop |
| < 768px | Single column, 1 camera at a time | Mobile (alerts only view recommended) |

The mobile view should prioritize the **alert feed** — operators on the floor check alerts on their phone, not camera grids.

---

## 10. Iconography

Use **Lucide Icons** (open source, consistent, works with React).

| Action | Icon | Notes |
|--------|------|-------|
| Camera | `camera` | |
| Alert | `alert-triangle` | |
| Settings | `settings` | |
| Zone | `pentagon` | |
| Acknowledge | `check` | |
| Expand | `maximize-2` | |
| Collapse | `minimize-2` | |
| Filter | `filter` | |
| Search | `search` | |
| Live indicator | `radio` (pulsing) | |
| Download/Export | `download` | |
| Full screen | `monitor` | |

---

## 11. UX Patterns

### 11.1 Alert Toast (Real-Time)

When a new P1/P2 alert fires:
1. Toast slides in from top-right
2. Contains: severity badge + camera name + violation type
3. Stays for 8 seconds (P1) or 5 seconds (P2/P3)
4. Click toast → navigate to alert detail
5. P1 alerts play a subtle audio chime (configurable, off by default)

### 11.2 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-9` | Switch to camera 1-9 in single view |
| `G` | Back to grid view |
| `A` | Open alert center |
| `D` | Open dashboard |
| `Space` | Acknowledge selected alert |
| `Esc` | Close modal / back to grid |
| `F` | Toggle fullscreen |

### 11.3 Empty States

Every list/grid should handle empty state gracefully:
- No cameras: "Add your first camera to get started" + Add Camera button
- No alerts: "All clear. No violations detected." + green checkmark
- No data for chart: "Not enough data yet. Violations will appear here." + subtle illustration

### 11.4 Error States

- Camera disconnected: Grey overlay on camera card with "Disconnected" label + reconnect spinner
- Inference error: Yellow banner at top of page
- License issue: Red banner, cannot be dismissed

---

## 12. Tech Stack (Frontend)

| Layer | Tech | Why |
|-------|------|-----|
| Framework | React 18+ | Already in use |
| Styling | Tailwind CSS | Utility-first, fast to iterate, easy dark theme |
| Charts | Recharts | Already in use, lightweight |
| Icons | Lucide React | Clean, consistent, tree-shakeable |
| State | Zustand | Simple, no boilerplate (replace Redux if using) |
| WebSocket | Native WebSocket | Already in use |
| Canvas | HTML5 Canvas API | For bounding box / zone overlays |
| Fonts | Inter + JetBrains Mono | Google Fonts or self-hosted |

---

## 13. Design Tokens (CSS Custom Properties)

```css
:root {
  /* Backgrounds */
  --bg-primary: #0F1117;
  --bg-secondary: #1A1D27;
  --bg-tertiary: #242836;
  --border: #2E3345;

  /* Text */
  --text-primary: #F0F2F5;
  --text-secondary: #8B92A5;
  --text-muted: #5A6178;

  /* Severity */
  --severity-critical: #FF3B3B;
  --severity-high: #FF8C00;
  --severity-medium: #FFD000;
  --severity-info: #3B82F6;
  --severity-ok: #22C55E;

  /* Accent */
  --accent-primary: #6366F1;
  --accent-hover: #818CF8;
  --accent-pressed: #4F46E5;

  /* Overlays */
  --zone-outline: #38BDF8;
  --detection-box: #A78BFA;

  /* Typography */
  --font-ui: 'Inter', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;

  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-8: 48px;

  /* Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-full: 9999px;
}
```
