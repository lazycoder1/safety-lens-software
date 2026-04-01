# SafetyLens — Feature Set & Status

**Date:** 2026-03-08

---

## Architecture

### Current (Demo — M1 Pro Dev Machine)

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (http://localhost:3030)                                 │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌───────────────┐   │
│  │ Live View  │ │ Alert Feed │ │Dashboard │ │ Camera Config │   │
│  │ (MJPEG)    │ │ (WebSocket)│ │(Charts)  │ │ (CRUD + YOLOe)│   │
│  └─────┬──────┘ └─────┬──────┘ └────┬─────┘ └───────┬───────┘   │
│        │              │             │                │           │
└────────┼──────────────┼─────────────┼────────────────┼───────────┘
         │              │             │                │
    MJPEG stream   WebSocket     REST API          REST API
         │              │             │                │
┌────────┼──────────────┼─────────────┼────────────────┼───────────┐
│  FastAPI Backend (http://localhost:8000)                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Video Processor Threads (1 per camera)                 │     │
│  │                                                         │     │
│  │  cam1 ──► YOLOv8n (trained) ──► draw boxes ──► MJPEG   │     │
│  │  cam2 ──► YOLOv8n (trained) ──► draw boxes ──► MJPEG   │     │
│  │  cam3 ──► YOLOe (open-vocab) ──► draw boxes ──► MJPEG  │     │
│  │           ▲                                             │     │
│  │           │ set_classes() on config change               │     │
│  │           │ (CPU round-trip for MobileCLIP)             │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Alert Engine │  │ Config Mgr   │  │ VLM Worker (Ollama)  │   │
│  │ violations + │  │ config.json  │  │ qwen3-vl:8b          │   │
│  │ cooldown +   │  │ camera CRUD  │  │ (disabled for now)   │   │
│  │ WS broadcast │  │ live reload  │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  Models loaded in memory:                                        │
│  ├── YOLOv8n trained (6MB) ──► MPS (Metal)                      │
│  ├── YOLOe-11s-seg (27MB) ──► MPS (Metal)                       │
│  └── MobileCLIP text encoder (572MB) ──► CPU (for set_classes)   │
│                                                                  │
│  Video source: .mp4 files in test-videos/ (looped)               │
└──────────────────────────────────────────────────────────────────┘
```

### End State (Production — DGX Spark or Jetson)

```
┌─────────────────────────────────────┐
│  IP Cameras (RTSP)                  │
│  cam1 ──┐                           │
│  cam2 ──┤                           │
│  cam3 ──┤  RTSP streams             │
│  ...    ├──────────────────────┐    │
│  cam50 ─┘                      │    │
└────────────────────────────────┼────┘
                                 │
┌────────────────────────────────┼──────────────────────────────────┐
│  DGX Spark / Jetson AGX                                          │
│                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Inference Engine (CUDA / TensorRT)                     │     │
│  │                                                         │     │
│  │  YOLO26n (TensorRT FP16)         YOLOe (TensorRT)      │     │
│  │  ├── helmet, vest, phone          ├── text prompts      │     │
│  │  ├── person, forklift             ├── visual prompts    │     │
│  │  ├── fall (pose model)            └── per-camera class  │     │
│  │  └── ~990 FPS capacity                 sets             │     │
│  │                                                         │     │
│  │  50 cameras @ 6fps = 300 fps needed = ~30% GPU          │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Alert Engine │  │ Zone Engine  │  │ Training Pipeline    │   │
│  │ per-rule     │  │ polygon ROI  │  │ YOLOe auto-label     │   │
│  │ cooldown +   │  │ per-zone     │  │ -> Roboflow review   │   │
│  │ escalation + │  │ rule assign  │  │ -> YOLO26 retrain    │   │
│  │ DB persist   │  │ shift-aware  │  │ -> hot-swap deploy   │   │
│  └──────┬───────┘  └──────────────┘  └──────────────────────┘   │
│         │                                                        │
│  ┌──────┴───────────────────────────────────────────────────┐   │
│  │  Notification Layer                                       │   │
│  │  Dashboard + Email + WhatsApp + Webhook + ERP/SAP         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  VLM Engine (on-prem, same device)                        │   │
│  │  qwen3-vl:8b via Ollama (DGX Spark has 128GB RAM)        │   │
│  │  ├── Gangway blockage reasoning                           │   │
│  │  ├── LOTO compliance check                                │   │
│  │  ├── Ambiguous item escalation                            │   │
│  │  └── "Why is this unsafe?" incident reasoning             │   │
│  │  YOLO-gated: only ~20% of frames reach VLM               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  nginx ──► React Dashboard (built static files)           │   │
│  │  ├── /live          Multi-camera grid + single cam view   │   │
│  │  ├── /alerts        Alert feed + acknowledge + history    │   │
│  │  ├── /dashboard     Live KPIs, heatmap, trends            │   │
│  │  ├── /configure     Cameras, zones, rules, YOLOe classes  │   │
│  │  └── /reports       Shift reports, PDF/CSV export         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Everything runs on-prem. No cloud dependency. No data leaves.   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow Per Frame

```
Camera frame (1080p, 6fps)
    │
    ├──► YOLO26 inference (~1ms TensorRT)
    │    ├── Detections: [{person, 0.92, bbox}, {hard_hat, 0.87, bbox}, ...]
    │    ├── Rule check: person in zone without helmet? phone detected?
    │    │   ├── No violation ──► annotate frame, stream to UI
    │    │   └── Violation ──► create alert, broadcast WS, notify
    │    └── Frame + detections stored for training pipeline
    │
    ├──► YOLOe inference (if camera uses open-vocab mode)
    │    ├── set_classes(["person", "hairnet", "gloves", ...])
    │    ├── Same flow as YOLO26 but with dynamic classes
    │    └── Over time: YOLOe labels -> train YOLO26 -> replace YOLOe
    │
    └──► VLM (every 30-60s, only if YOLO flags something or periodic scan)
         ├── Frame -> base64 -> Ollama on-prem (qwen3-vl:8b)
         ├── Prompt: "Is the gangway clear? Any safety hazards?"
         └── Response parsed for violation keywords -> alert if found

All processing on-prem. No data leaves the device. No internet required.
```

---

## What We Have Now (Working Demo)

### Backend
| Component | Status | Details |
|-----------|--------|---------|
| FastAPI server | Done | Multi-camera video processing, MJPEG streaming |
| YOLO inference (YOLOv8n trained) | Done | 4 classes: person, hard_hat, safety_vest, equipment |
| YOLOe open-vocab detection | Done | Text-prompt classes, live-editable from frontend |
| Multi-camera threading | Done | Each camera runs in its own thread with stop/restart |
| Alert engine | Done | Violation detection, cooldown, severity levels (P1-P4) |
| WebSocket alert push | Done | Real-time alerts to frontend |
| Camera CRUD API | Done | Add/edit/delete cameras, change demo mode, update classes |
| Config persistence | Done | JSON-based config, survives restart |
| VLM integration (Ollama) | Done | qwen3-vl:8b via Ollama API (currently disabled to save RAM) |

### Frontend
| Page | Status | Details |
|------|--------|---------|
| `/live` — Live View | Done | Multi-camera grid, click-to-maximize, MJPEG streams |
| `/live?cam=X` — Single Camera | Done | Full-screen single camera with YOLOe class badges |
| `/alerts` — Alert Center | Done | Alert list with severity, acknowledge, filter |
| `/dashboard` — Dashboard | Done | KPIs, compliance heatmap, violation trend charts (mock data) |
| `/configure/cameras` — Camera Config | Done | Add/edit/delete cameras, demo mode selector, YOLOe class editor |
| `/configure/rules` — Detection Rules | Done | Rule list UI (static) |
| `/configure/alerts` — Alert Routing | Done | Routing config UI (static) |
| `/system/settings` — System Settings | Done | Global config UI |
| `/reports` — Reports | Placeholder | Not built |
| `/configure/zones` — Zone Management | Placeholder | Not built |

### Detection Modes Working
| Mode | How It Works | Cameras Using It |
|------|-------------|-----------------|
| `yolo` | Trained YOLOv8n, fixed 4 classes | cam1, cam2 |
| `yoloe` | Open-vocab, text-prompt classes, live-editable | cam3 |
| `yolo+vlm` | YOLO + periodic VLM scene analysis | Disabled (saves RAM) |

### Models on Disk
| Model | File | Size | Status |
|-------|------|------|--------|
| YOLOv8n (trained) | `runs/industrial_safety_yolov8n/weights/best.pt` | ~6MB | In use |
| YOLOe-11s-seg | `yoloe-11s-seg.pt` | 27MB | In use |
| YOLO26n | `yolo26n.pt` | 5.3MB | Downloaded, needs training |
| MobileCLIP text encoder | `backend/mobileclip_blt.ts` | 572MB | Auto-downloaded, used by YOLOe |

---

## End State (Production for TMEIC)

### Detection Rules — Basic Package (Tier 1)

7 rules from 5 types of customer footage. No staging needed — normal working day recordings.

| # | Rule | Detection Method | Alert Type | Status |
|---|------|-----------------|------------|--------|
| 1 | Helmet not worn in safety zone | YOLO26 + ROI polygon | P2 — real-time | Partially built (YOLOv8n, no ROI UI yet) |
| 2 | Safety vest not worn | YOLO26 | P2 — real-time | Working (YOLOv8n) |
| 3 | Forklift operator without helmet | YOLO26 | P2 — real-time | Needs forklift-specific training data |
| 4 | Mobile phone usage while working | YOLO26 (phone is COCO class) | P3 — real-time | Needs training on customer footage |
| 5 | Zone intrusion — person in restricted area | YOLO26 + geofence polygon | P1 — real-time | Needs zone drawing UI |
| 6 | Gangway / aisle blockage | VLM periodic scan | P3 — periodic | VLM integration done, needs tuning |
| 7 | Person detection + headcount per zone | YOLO26 | P4 — info | Working (person class) |

### Platform Features — End State

#### Core Engine
| Feature | Now | End State | Gap |
|---------|-----|-----------|-----|
| YOLO26 trained model | YOLOv8n (4 classes) | YOLO26n trained on customer data | Train YOLO26 on customer footage |
| YOLOe open-vocab | Working (text prompts) | Working + visual prompts | Add visual prompt support in UI |
| VLM scene analysis | Working (Ollama local) | Ollama on DGX Spark (qwen3-vl:8b) | Deploy Ollama on Spark |
| Multi-camera | 3 cameras demo | 20-50 cameras | Thread pool, batched inference |
| Inference device | MPS (M1 Pro dev) | CUDA/TensorRT (DGX Spark or Jetson) | TensorRT export |

#### Alert System
| Feature | Now | End State | Gap |
|---------|-----|-----------|-----|
| Alert detection | Cooldown-based, per-camera | Per-rule cooldown, dedup, escalation | Build escalation logic |
| Alert delivery | Dashboard + WebSocket toast | + Email + WhatsApp + webhook | Integrate notification APIs |
| Alert acknowledge | Manual in UI | + Auto-resolve, snooze, escalation chains | Build workflow engine |
| Alert history | In-memory deque (200 max) | Database (SQLite/Postgres) | Add DB persistence |

#### Zone Management
| Feature | Now | End State | Gap |
|---------|-----|-----------|-----|
| Zone definition | Hardcoded per camera | Polygon ROI drawn in UI on camera feed | Build zone drawing UI |
| Zone rules | Global rules | Per-zone rule assignment | Link rules to zones |
| Zone scheduling | None | Shift-aware (day/evening/night rules) | Build schedule config |

#### Dashboard & Reporting
| Feature | Now | End State | Gap |
|---------|-----|-----------|-----|
| KPI cards | Mock data | Live from alert DB | Connect to real data |
| Compliance heatmap | Mock data | Live zone x shift matrix | Connect to real data |
| Violation trends | Mock data | Live 24h rolling chart | Connect to real data |
| Shift reports | None | Auto-generated per shift | Build report generator |
| PDF/CSV export | None | Compliance audit export | Build export |

#### Training Pipeline
| Feature | Now | End State | Gap |
|---------|-----|-----------|-----|
| Frame extraction | Script (`01_extract_frames.py`) | Automated from live cameras | Schedule extraction |
| Auto-labeling | Script (`02_autolabel_yolo_world.py`) | YOLOe auto-label pipeline | Update script to use YOLOe |
| Human review | Manual | Roboflow integration | Setup Roboflow project |
| Model training | Script (`03_train_yolo.py`) | One-click retrain in UI | Build training trigger |
| Model deploy | Manual file swap | Hot-swap trained model | Build model management |

---

## Workflow: Camera Setup to Live Detection

### Current (Demo)
```
1. Drop .mp4 file in test-videos/
2. Add camera via UI (name, video file, demo mode, classes)
3. Backend starts processing thread
4. Live view shows detections + alerts fire
5. Edit YOLOe classes from UI -> model updates live
```

### End State (Production)
```
1. Customer installs cameras, connects RTSP streams
2. Admin adds camera in UI (name, RTSP URL, zone, rules)
3. Zone polygon drawn on camera feed in UI
4. Rules assigned per zone (helmet required, no phone, etc.)
5. System runs YOLO26 on every frame, fires alerts
6. For open-vocab needs, YOLOe text/visual prompts configured per camera
7. VLM periodic scan for scene reasoning (gangway, compliance)
8. Alerts push to dashboard + email/WhatsApp
9. Weekly: auto-extract hard frames -> review -> retrain -> deploy
```

### Workflow: Adding a New Detection Rule (End State)

**If object is standard (helmet, vest, phone, person):**
```
Already in YOLO26 trained model -> add rule logic in UI -> done
```

**If object is novel (hairnet, harness, specific PPE):**
```
1. Add YOLOe text prompt (e.g., "safety harness") -> works immediately
2. YOLOe auto-labels frames over days/weeks
3. Human reviews labels in Roboflow
4. Train YOLO26 with new class -> deploy
5. Switch from YOLOe to YOLO26 for that class (faster, more accurate)
```

**If rule needs scene reasoning (gangway blocked, LOTO compliance):**
```
VLM periodic scan with custom prompt -> alert on violation keywords
```

---

## What Needs Building (Priority Order)

### P0 — Required for First Demo with Customer Footage
1. Train YOLO26 on customer footage (replace YOLOv8n)
2. RTSP stream support (replace .mp4 file input)
3. Alert DB persistence (SQLite — replace in-memory deque)

### P1 — Required for Tier 1 Delivery
4. Zone drawing UI (polygon ROI on camera feed)
5. Per-zone rule assignment
6. Email/webhook alert delivery
7. Live dashboard connected to real alert data
8. TensorRT export for DGX Spark / Jetson

### P2 — Tier 2 Features
9. YOLOe visual prompt support in UI
10. Auto-labeling pipeline (YOLOe -> Roboflow)
11. Model retraining trigger from UI
12. Shift-aware scheduling
13. WhatsApp/SMS integration
14. PDF/CSV compliance reports

### P3 — Tier 3 Features
15. VLM on-prem optimization (larger model, faster inference on Spark)
16. Escalation chains + auto-resolve
17. ERP/SAP webhook integration
18. Continuous auto-improve pipeline
19. Multi-site management
