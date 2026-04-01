# SafetyLens — Frontend Architecture

**Version:** 1.0
**Date:** 2026-03-23
**Design Inspiration:** Linear.app

---

## 1. Design Philosophy

Borrowed from Linear — adapted for a control room:

- **Speed is a feature** — every interaction < 50ms. Optimistic UI everywhere.
- **Quiet when nominal** — the UI is calm when everything is fine. Color and motion appear only for deviations. ("Dark cockpit" philosophy from aviation.)
- **Keyboard-first** — operators shouldn't need a mouse for common actions.
- **Strip the chrome** — no decorative borders, shadows, or ornaments. Every element earns its place.
- **Dense but scannable** — more info per viewport, tight padding, monospace for data.

---

## 2. Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | React 18+ with Vite | Fast dev, HMR, lazy routes |
| Styling | Tailwind CSS + CSS custom properties | Utility-first, dark theme tokens |
| State (client) | Zustand (one store per feature) | Minimal boilerplate, selector-based subscriptions |
| State (server) | TanStack Query | Caching, background refetch for config/CRUD |
| WebSocket | Singleton manager outside React → Zustand | High-frequency updates don't trigger React reconciler |
| Command Palette | cmdk (by Linear's team) | Fuzzy search, keyboard nav, accessible |
| Charts | Recharts or Tremor | Already in use / Tailwind-native |
| Icons | Lucide React | Clean, consistent, tree-shakeable |
| Animations | Framer Motion + CSS transitions | Layout animations, exit animations, springs |
| Toasts | sonner | Stacking, severity, auto-dismiss |
| Virtualization | @tanstack/react-virtual | Alert lists with 1000+ items |
| Video | Native `<img>` for MJPEG + `<canvas>` overlay | Browser handles MJPEG natively |
| UI Primitives | Radix UI (via shadcn/ui) | Accessible, unstyled, composable |

---

## 3. Folder Structure (Feature-Based)

```
src/
├── app/                              # App shell
│   ├── App.tsx                       # Root component
│   ├── routes.tsx                    # Lazy-loaded route definitions
│   ├── providers.tsx                 # Composed providers + WS init
│   └── layouts/
│       └── DashboardLayout.tsx       # Top bar + content area
│
├── features/                         # Feature modules (the core)
│   ├── video-grid/                   # Live camera grid
│   │   ├── components/
│   │   │   ├── VideoGrid.tsx         # Grid container (2x2, 3x3, 4x4)
│   │   │   ├── VideoCell.tsx         # Single camera card
│   │   │   ├── MjpegStream.tsx       # MJPEG <img> wrapper
│   │   │   ├── CanvasOverlay.tsx     # Bounding box + zone overlay
│   │   │   └── SingleCameraView.tsx  # Expanded single-camera view
│   │   ├── hooks/
│   │   │   ├── useGridLayout.ts      # Grid size persistence
│   │   │   └── useCanvasOverlay.ts   # Draw detections on canvas
│   │   ├── stores/
│   │   │   └── videoGridStore.ts     # Overlays, grid state
│   │   ├── types.ts
│   │   └── index.ts                  # Public API
│   │
│   ├── alerts/                       # Alert center
│   │   ├── components/
│   │   │   ├── AlertPanel.tsx        # Full alert page
│   │   │   ├── AlertCard.tsx         # Single alert card
│   │   │   ├── AlertFilters.tsx      # Severity/camera/time filters
│   │   │   ├── AlertToast.tsx        # Real-time toast notification
│   │   │   └── AlertInbox.tsx        # Linear-style inbox sidebar
│   │   ├── hooks/
│   │   │   └── useAlerts.ts          # Alert subscription + actions
│   │   ├── stores/
│   │   │   └── alertStore.ts         # Alert state (Map for O(1) updates)
│   │   ├── types.ts
│   │   └── index.ts
│   │
│   ├── dashboard/                    # KPI dashboard
│   │   ├── components/
│   │   │   ├── DashboardPage.tsx     # Dashboard layout
│   │   │   ├── KpiCard.tsx           # Big number + trend
│   │   │   ├── ViolationTrend.tsx    # 24h line chart
│   │   │   ├── ComplianceHeatmap.tsx # Zone x shift matrix
│   │   │   └── TopViolations.tsx     # Bar chart by camera
│   │   ├── hooks/
│   │   │   └── useDashboardData.ts   # Aggregated data
│   │   ├── stores/
│   │   │   └── dashboardStore.ts
│   │   └── index.ts
│   │
│   ├── zones/                        # Zone management
│   │   ├── components/
│   │   │   ├── ZoneEditor.tsx        # Polygon drawing on camera feed
│   │   │   ├── ZoneList.tsx          # List of zones per camera
│   │   │   └── ZoneRuleAssign.tsx    # Assign rules to zones
│   │   ├── hooks/
│   │   │   └── useZoneDrawing.ts     # Canvas polygon drawing logic
│   │   ├── stores/
│   │   │   └── zoneStore.ts
│   │   └── index.ts
│   │
│   ├── config/                       # Settings & configuration
│   │   ├── components/
│   │   │   ├── CameraConfig.tsx      # Camera CRUD
│   │   │   ├── DetectionConfig.tsx   # Enable/disable detection modules
│   │   │   ├── NotificationConfig.tsx # Alert routing settings
│   │   │   └── SystemSettings.tsx    # Global settings
│   │   ├── hooks/
│   │   │   └── useConfig.ts
│   │   └── index.ts
│   │
│   ├── command-palette/              # Cmd+K
│   │   ├── components/
│   │   │   └── CommandPalette.tsx    # cmdk-based palette
│   │   ├── hooks/
│   │   │   └── useCommands.ts        # Command registry
│   │   ├── stores/
│   │   │   └── commandStore.ts
│   │   └── index.ts
│   │
│   └── license/                      # License activation
│       ├── components/
│       │   ├── ActivationWizard.tsx  # First-boot activation screen
│       │   ├── LicenseStatus.tsx     # License info in settings
│       │   └── LicenseExpiry.tsx     # Expiry warning banner
│       └── index.ts
│
├── shared/                           # Cross-feature shared code
│   ├── components/                   # Generic UI primitives
│   │   ├── Button.tsx
│   │   ├── Badge.tsx
│   │   ├── Dialog.tsx
│   │   ├── Tooltip.tsx
│   │   ├── Skeleton.tsx              # Shimmer loading placeholder
│   │   ├── StatusDot.tsx             # Animated status indicator
│   │   └── SeverityBadge.tsx         # P1/P2/P3/P4 pill
│   ├── hooks/
│   │   ├── useWebSocket.ts           # React bridge to WS manager
│   │   ├── useKeyboardShortcut.ts    # Global shortcut registry
│   │   └── useReducedMotion.ts       # Accessibility
│   ├── lib/
│   │   ├── ws.ts                     # WebSocket singleton manager
│   │   ├── api.ts                    # REST API client (fetch wrapper)
│   │   ├── utils.ts                  # Formatters, helpers
│   │   └── animations.ts            # Shared timing/easing constants
│   ├── stores/
│   │   ├── connectionStore.ts        # WS connection state
│   │   └── uiStore.ts               # Theme, sidebar, layout prefs
│   └── types/
│       ├── ws-messages.ts            # WebSocket message types
│       ├── camera.ts                 # Camera types
│       └── alert.ts                  # Alert types
│
├── styles/
│   ├── globals.css                   # Tailwind directives + base styles
│   └── theme.css                     # CSS custom property tokens
│
└── main.tsx                          # Entry point
```

**Rules:**
- Features never import from other features directly — they communicate through shared stores.
- Each feature's `index.ts` is the public API — only export what others need.
- `shared/` is for truly generic, feature-agnostic code.

---

## 4. State Architecture

```
                    ┌──────────────────────────┐
                    │   WebSocket Manager      │
                    │   (singleton, outside     │
                    │    React tree)            │
                    └──────────┬───────────────┘
                               │ batched messages (30fps)
                               ▼
              ┌────────────────┬────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │  alertStore    │ │ videoGrid   │ │ dashboard   │
    │  (Zustand)     │ │ Store       │ │ Store       │
    │                │ │ (Zustand)   │ │ (Zustand)   │
    └─────────┬──────┘ └──────┬──────┘ └──────┬──────┘
              │                │                │
              │  selector      │  selector      │  selector
              ▼                ▼                ▼
    ┌─────────────────────────────────────────────────────┐
    │              React Components                        │
    │  (only re-render when their subscribed slice changes) │
    └─────────────────────────────────────────────────────┘

    Config/CRUD data:
    ┌──────────────────┐
    │  TanStack Query  │ ──► REST API ──► FastAPI
    └──────────────────┘
```

**Key pattern:** The WebSocket manager lives outside React. It buffers messages and flushes them to Zustand stores at ~30fps. React components subscribe to specific slices via selectors, so one camera getting detections doesn't re-render the other 9 cells.

---

## 5. WebSocket Architecture

```typescript
// shared/lib/ws.ts — singleton, no React dependency

class WebSocketManager {
  private ws: WebSocket | null = null;
  private handlers = new Map<string, Set<Function>>();
  private buffer: any[] = [];

  connect(url: string) { ... }

  // Buffer messages, flush at 30fps
  private startFlush() {
    setInterval(() => {
      if (this.buffer.length === 0) return;
      const batch = this.buffer.splice(0);
      // Group by type, dispatch to handlers
      for (const [type, payloads] of groupBy(batch, 'type')) {
        this.handlers.get(type)?.forEach(fn => fn(payloads));
      }
    }, 33); // ~30fps
  }

  subscribe(type: string, handler: Function) { ... }
  send(type: string, payload: any) { ... }
}

export const wsManager = new WebSocketManager();
```

**Bridge to stores (app startup):**
```typescript
wsManager.subscribe('alerts', (batch) => alertStore.getState().bulkUpdate(batch));
wsManager.subscribe('detections', (batch) => videoGridStore.getState().updateOverlays(batch));
wsManager.subscribe('metrics', (batch) => dashboardStore.getState().appendMetrics(batch));
```

---

## 6. Video Grid + Canvas Overlay

```
┌──────────────────────────────────────────┐
│  VideoCell                                │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │  <img src="/api/streams/cam1/mjpeg">│ │  ← browser handles MJPEG natively
│  │                                      │ │
│  │  ┌──────────────────────────────┐   │ │
│  │  │  <canvas> (absolute overlay) │   │ │  ← bboxes + zones drawn here
│  │  │  pointer-events: none        │   │ │
│  │  └──────────────────────────────┘   │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  CAM-01 · Production Floor    ● LIVE     │  ← status bar
└──────────────────────────────────────────┘
```

- `<img>` natively streams MJPEG — no custom JS frame decoding
- `<canvas>` sits on top with `position: absolute; pointer-events: none`
- Each `CanvasOverlay` subscribes to ONLY its camera's detections via Zustand selector
- Canvas redraws via `requestAnimationFrame`
- Only animates `transform` and `opacity` — GPU composited

---

## 7. Command Palette (Cmd+K)

Using `cmdk` (created by Linear's team):

```
┌─────────────────────────────────────────────────┐
│  🔍 Search cameras, alerts, actions...           │
├─────────────────────────────────────────────────┤
│  CAMERAS                                         │
│  ├─ CAM-01 · Production Floor       ● Online    │
│  ├─ CAM-03 · Assembly Line          ● Alert     │
│  └─ CAM-05 · Loading Dock           ● Online    │
│                                                   │
│  ACTIONS                                          │
│  ├─ Acknowledge all alerts              Space    │
│  ├─ Switch to 3x3 grid                 ⌘3       │
│  └─ Export compliance report            ⌘⇧E      │
│                                                   │
│  ZONES                                            │
│  ├─ Assembly Line                                │
│  └─ Motor Bay (restricted)                       │
└─────────────────────────────────────────────────┘
```

Entry: scale from 95% + fade in (150ms). Backdrop blur.

---

## 8. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Cmd+K` | Open command palette |
| `1-9` | Switch to camera 1-9 |
| `G` | Back to grid view |
| `A` | Open alert center |
| `D` | Open dashboard |
| `Space` | Acknowledge selected alert |
| `J / K` | Navigate alert list (vim-style) |
| `Enter` | Expand selected camera / open alert |
| `Esc` | Close modal / back to grid |
| `F` | Toggle fullscreen |
| `Cmd+1/2/3/4` | Switch grid layout (1x1, 2x2, 3x3, 4x4) |

---

## 9. Animation Constants

```typescript
// shared/lib/animations.ts

// Linear-inspired: fast, small, precise
export const timing = {
  instant: 0.1,      // button press feedback
  fast: 0.15,        // most transitions
  normal: 0.2,       // page transitions
  smooth: 0.3,       // modal open/close
};

export const easing = {
  default: [0.25, 0.1, 0.25, 1.0],    // fast in, gentle out
  enter: [0.0, 0.0, 0.2, 1.0],
  exit: [0.4, 0.0, 1.0, 1.0],
};

export const springs = {
  snappy: { type: "spring", stiffness: 500, damping: 30 },
  responsive: { type: "spring", stiffness: 300, damping: 25 },
  gentle: { type: "spring", stiffness: 200, damping: 20 },
};

// Rule: only animate transform + opacity. Never width/height/top/left.
// Rule: shift distances 4-12px max. Large moves feel sluggish.
```

---

## 10. Component Patterns

### Alert Card — Linear Style
```
┌─────────────────────────────────────────────────────────┐
│  ●  P2 · No Helmet Detected                    2m ago  │
│     CAM-03 · Assembly Line                         ✓    │
└─────────────────────────────────────────────────────────┘
  ↑                                                   ↑
  3px left border                              acknowledge
  severity color                               (1-click)
```

- Left border: severity color
- Unread: slightly brighter bg
- Hover: reveal actions (acknowledge, jump to camera)
- New item animation: fade in + slide up 8px (150ms)
- Critical: background pulse (red 15% → 0%, 1.5s loop)

### KPI Card
```
┌────────────────────────┐
│  Violations Today      │  ← --text-sm, secondary
│                        │
│       23               │  ← 48px, JetBrains Mono
│                        │
│  ▼ 12% vs yesterday   │  ← green (fewer = good)
└────────────────────────┘
```

### Status Dot (Linear-inspired)
- **Online**: static green dot
- **Recording/Active**: green dot with rotating arc animation (Linear's "In Progress" pattern)
- **Alert**: red dot, pulsing (1.5s)
- **Offline**: hollow gray dot
- **Degraded**: yellow dot, slow pulse

---

## 11. Theme Tokens (CSS Custom Properties)

```css
:root {
  /* Linear-inspired dark theme */
  --bg-base: #0F1117;
  --bg-surface: #1A1D27;
  --bg-elevated: #242836;
  --bg-hover: rgba(255, 255, 255, 0.04);
  --bg-active: rgba(255, 255, 255, 0.06);

  --text-primary: #E8E8ED;
  --text-secondary: #8B92A5;
  --text-muted: #555566;

  --border-subtle: rgba(255, 255, 255, 0.06);
  --border-default: rgba(255, 255, 255, 0.08);

  /* Severity */
  --severity-critical: #FF3B3B;
  --severity-high: #FF8C00;
  --severity-medium: #FFD000;
  --severity-info: #3B82F6;
  --severity-ok: #22C55E;

  /* Accent */
  --accent: #6366F1;
  --accent-hover: #818CF8;

  /* Overlays */
  --zone-outline: #38BDF8;
  --detection-box: #A78BFA;

  /* Typography */
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body {
  -webkit-font-smoothing: antialiased;
  font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv09';
}
```

---

## 12. Performance Rules

1. **WebSocket data → Zustand stores (outside React)** — never setState directly from WS
2. **Canvas for video overlays** — detection bounding boxes go through canvas, not React DOM
3. **Zustand selectors** — components subscribe to slices, not entire stores
4. **Virtualize alert lists** — @tanstack/react-virtual for 1000+ items
5. **Only animate transform + opacity** — GPU composited, 60fps
6. **Skeleton screens, not spinners** — CSS shimmer, no JS
7. **Lazy route splitting** — each page is a separate chunk
8. **MJPEG via native `<img>`** — browser handles the stream, no JS decode

---

## 13. Critical UX Patterns

### Alert-Driven Auto-Focus
When a P1 alert fires, the relevant camera auto-promotes to the spotlight position in the grid. The operator doesn't need to search — the system brings the problem to them.

### Side Panel (Linear's Peek View)
Clicking a camera tile opens a right-side detail panel (~40% width) without leaving the grid. Shows: enlarged feed, alert history, zone list, detection module status. Clicking another tile swaps the panel content — no navigation.

### Glanceable Status Bar
Top of every page — a persistent bar showing: zone statuses (colored dots), active alert count (badge), cameras online count. Visible from 3 meters.

### Empty States
- No cameras: "Add your first camera to get started" + Add Camera button
- No alerts: "All clear. No violations detected." + green checkmark
- No chart data: "Not enough data yet." + subtle placeholder
