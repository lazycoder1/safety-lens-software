# SafetyLens Base Pack — Requirements Document

**Last updated:** 2026-03-28
**Audience:** Engineers joining the SafetyLens project
**Status:** Active development

---

## What Is the Base Pack?

SafetyLens Base Pack is the entry-level product Techser sells to factories. It is a software-only license priced at INR 5,00,000. The customer provides their own hardware and IP cameras. Everything runs 100% on-premise — no cloud dependency, no data leaves the site.

**Scope:** Up to 10 IP cameras, 4 standard safety detections, real-time alerts, web dashboard.

---

## What the Customer Gets

### Cameras
- Up to 10 IP cameras via RTSP
- 1080p resolution, processed at 5 FPS per camera
- Auto-reconnect on connection drop (5-second retry)

### Detections (4 standard)

| # | Detection | What It Does |
|---|-----------|-------------|
| 1 | Person / Zone Intrusion | Alert when a person enters a restricted zone (drawn polygon) |
| 2 | Mobile Phone Usage | Detect phone in hand inside work areas |
| 3 | Animal Intrusion | Dog or cat detected on factory premises |
| 4 | PPE — Helmet + Safety Vest | Missing hard hat or missing hi-vis vest |

### Dashboard
- Live multi-camera grid with detection bounding box overlays
- Real-time alert feed via WebSocket
- KPI cards and charts (alert counts, compliance metrics)

### Alerts
- **Telegram:** Snapshot image with bounding box, camera name, timestamp. Target latency under 3 seconds.
- **Cooldown/dedup:** Per-rule cooldown with severity multipliers (P1=60s, P3=120s, P4=300s). Prevents alert fatigue.
- **Severity levels:** P1 through P4.
- **Alert lifecycle:** Acknowledge, resolve, snooze, mark false positive.

### Zone Management
- Draw polygon zones on camera feed from the web UI
- Assign detection rules per zone
- Zone enforcement runs inside the inference loop

### Configuration
- All settings managed from the web UI
- Persists across restarts (atomic JSON writes to `config.json`)
- Per-camera config: RTSP URL, zone assignment, detection mode, YOLOE classes

---

## Architecture Overview

```
IP Cameras (RTSP, 1080p, 5 FPS)
    |
    v
FastAPI Backend (Python, single process, multi-threaded)
    +-- 1 thread per camera (video_processor)
    +-- YOLO26n inference (COCO 80 classes)
    +-- YOLOE-11s inference (open-vocab PPE)
    +-- Zone enforcement (point-in-polygon)
    +-- Alert creation --> PostgreSQL
    +-- Telegram notification (fire-and-forget)
    +-- MJPEG stream encoding
    +-- WebSocket alert broadcast
    |
    v
React Frontend (served as static files by FastAPI)
    +-- MJPEG <img> tags for live video
    +-- WebSocket for real-time alerts
    +-- REST API for config/alerts/cameras
    +-- Zustand store for alert state
```

**Key design decisions:**
- Single Python process with one thread per camera. No microservices.
- MJPEG streams encoded server-side with bounding boxes drawn on frames.
- Frontend is a React SPA bundled and served by FastAPI (no separate web server in production).
- PostgreSQL for alert persistence, `config.json` for system/camera/rule config.
- Docker Compose for production deployment (multi-stage Dockerfile: Node build + CUDA runtime).

---

## Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLO26n | `yolo26n.pt` | 5.3 MB | COCO 80-class detection — person (class 0), cell phone (class 67), cat (class 15), dog (class 16) |
| YOLOE-11s-seg | `yoloe-11s-seg.pt` | 27 MB | Open-vocabulary detection via text prompts — helmet, safety vest (and other PPE like hairnet, gloves) |

Model weights are **not** in the git repo. Download separately or copy from model storage.

---

## Hardware Requirements

| Tier | Hardware | Approx. Cost | Notes |
|------|----------|-------------|-------|
| Demo / PoC | Desktop PC with RTX 3050 6GB | ~INR 45,000 | Handles 10 cameras at ~45% GPU. PyTorch inference is fine here. |
| Production | Jetson Orin NX 16GB | ~INR 1,00,000 | **Requires TensorRT FP16 export** (not built yet). Without TensorRT, cannot sustain 10 cameras at 5 FPS. |

---

## Key Config Reference

All configuration lives in `backend/config.json` (example at `backend/config.example.json`).

| Key | Description | Default |
|-----|-------------|---------|
| `database.url` | PostgreSQL connection string | — |
| `global.target_fps` | Frames per second per camera | 6 |
| `global.yolo_conf` | YOLO confidence threshold | 0.35 |
| `global.alert_cooldown` | Seconds between duplicate alerts | 60 |
| `cameras.*` | Per-camera: name, RTSP URL, zone, rules, YOLOE classes | — |
| `telegram.*` | Bot token, chat ID, severity filter | — |

---

## Build Status

### Done — Working in Code

| Feature | Implementation | Key Files |
|---------|---------------|-----------|
| RTSP camera input | OpenCV VideoCapture, auto-reconnect on drop (5s retry) | `backend/server.py` lines 538-681 |
| Person detection | YOLO26n COCO class 0 "person" | `backend/server.py`, `yolo26n.pt` |
| Zone intrusion | Point-in-polygon check on detection center vs zone polygon | `backend/server.py` lines 314-376 |
| Mobile phone detection | YOLO26n COCO class 67 "cell phone" | `backend/server.py` lines 392-404 |
| Animal intrusion | YOLO26n COCO classes 15 "cat", 16 "dog" | `backend/server.py` lines 393-422 |
| PPE (helmet + vest) | YOLOE-11s-seg open-vocabulary with text prompts | `backend/server.py` lines 286-297, `yoloe-11s-seg.pt` |
| Web dashboard | React frontend, MJPEG streams, WebSocket alerts | `frontend/src/pages/LiveView.tsx`, `AlertCenter.tsx`, `Dashboard.tsx` |
| Detection overlays | Bounding boxes + labels drawn on MJPEG frames server-side | `backend/server.py` lines 190-272 |
| Telegram alerts | Photo + caption via Telegram Bot API, fire-and-forget | `backend/telegram_notifier.py` |
| Alert persistence | PostgreSQL with ThreadedConnectionPool, full CRUD | `backend/alert_store.py` |
| Cooldown / dedup | Per-rule cooldown with severity multipliers | `backend/server.py` lines 553-639 |
| Zone drawing UI | Canvas polygon drawing on camera feed, CRUD API | `frontend/src/pages/ZoneManagement.tsx`, `server.py` lines 1068-1128 |
| Zone enforcement | Polygon check in inference loop, creates P1/P2 alerts | `backend/server.py` lines 335-376 |
| Config persistence | Atomic JSON writes, thread-safe | `backend/config_manager.py` |
| Docker deployment | Multi-stage Dockerfile (Node build + CUDA runtime), docker-compose with Postgres | `Dockerfile`, `docker-compose.yml` |
| Camera management UI | Add/edit/delete cameras, RTSP URL, zone, detection mode | `frontend/src/pages/CameraConfig.tsx` |
| Alert management UI | Filter by severity/status, acknowledge, snooze, resolve, false positive | `frontend/src/pages/AlertCenter.tsx` |
| System settings UI | FPS, confidence, JPEG quality, cooldown, Telegram config | `frontend/src/pages/SystemSettings.tsx` |
| Alert routing UI | Severity x channel matrix (Telegram, email, webhook) | `frontend/src/pages/AlertRouting.tsx` |

### Partial — UI Exists, Backend Gaps

| Feature | What Works | What's Missing |
|---------|-----------|---------------|
| Alert routing | UI matrix for severity x channel exists and renders | Only Telegram actually sends. Email, WhatsApp, webhook channels are UI-only — no backend implementation. |
| Detection rules | 12 preset rules in config, enable/disable toggle works | Custom rule creation from UI does not persist to backend. The WHEN/IF/THEN rules engine UI is a frontend mock only. |
| Dashboard KPIs | Charts and cards render with real alert data | Some KPI calculations may use mock data. Compliance percentage needs real calculation logic. |

### Not Built — Needed for Production

| Feature | Why It Matters | Effort Estimate |
|---------|---------------|-----------------|
| **User management / RBAC** | No login, no roles, no permissions. Anyone on the network can change config, delete alerts, modify rules. Operators should not be able to edit rules; integrators should not need to acknowledge alerts during setup. No auth of any kind exists — no UI, no backend. | 5-7 days |
| **Feature gating (greyed-out disabled states)** | Unlicensed features should show as disabled with "Contact vendor to upgrade" message. Currently all features are always visible and fully functional regardless of license. No visual distinction between licensed and unlicensed modules. No UI or backend for this. | 3-5 days (alongside license work) |
| **TensorRT FP16 export** | PyTorch-only inference right now. Jetson Orin NX needs TensorRT for 3-5x speedup to sustain 10 cameras at 5 FPS. Desktop RTX 3050 works without it. | 2-3 days |
| **License validation** | No license check on startup. Anyone can run the software. Need signed `.lic` file verification (Ed25519 public key, camera count enforcement, feature gating). | 3-5 days |
| **Camera count enforcement** | No hardcoded limit. Config accepts any number of cameras. License should enforce the 10-camera cap. | Included in license work |
| **Health monitoring** | No CPU/GPU/RAM visibility. No camera connection status beyond logs. Operators need a system health view. | 3-5 days |
| **Auto-start on boot** | No systemd service file. Docker handles restart policies but there is no clean first-boot setup flow. | 1 day |
| **Email alerts** | Only Telegram works. Email channel is shown in the routing UI but backend sends nothing. | 2 days |
| **WhatsApp alerts** | Shown in alert routing UI but not functional. | 2 days |
| **Webhook / API alerts** | JSON POST to external URL for ERP/SCADA integration. Not built. | 1-2 days |
| **PDF / CSV export** | Shift reports, compliance reports. Dashboard shows data but has no export functionality. | 3-5 days |
| **Detection rules DB persistence** | Rules live in `config.json`, not PostgreSQL. Works for now but does not scale for the rules engine. | 2-3 days |
| **Audit log** | No record of who changed what. Required for ISO 45001 safety audits at factories. No UI, no backend. | 3-5 days |

---

## Running Locally

```bash
# Backend
cd backend
pip install -r ../requirements.txt
python server.py  # Starts on :8000

# Frontend (dev mode)
cd frontend
npm install
npm run dev  # Starts on :3030, proxies API to :8000

# Docker (production-like)
docker compose up --build  # Everything on :8000
```

---

## Priority — What to Work On Next

Listed in order of priority. Items 1-3 are blockers for shipping to customers.

1. **TensorRT FP16 export** — Blocks production deployment on Jetson hardware. Without this, the product only runs on desktop GPU.
2. **License validation + feature gating** — Blocks selling the product. No way to enforce paid licenses, camera limits, or disable unlicensed features. Feature gating needs greyed-out UI states for locked modules.
3. **User management / RBAC** — Blocks production deployment. Cannot ship a product where anyone on the network has full admin access.
4. **Rules engine backend** — Connect the WHEN/IF/THEN UI to actual rule evaluation and persistence.
5. **Email + webhook alerts** — Customers expect notification channels beyond Telegram.
6. **Health monitoring** — Operators need visibility into GPU utilization, camera status, and system health.
7. **Audit log** — Required for factory safety compliance (ISO 45001).
