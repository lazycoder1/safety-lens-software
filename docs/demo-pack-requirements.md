# SafetyLens Demo Pack — Requirements Document

**Last updated:** 2026-03-28
**Audience:** Engineers joining the SafetyLens project
**Status:** Partially built — 6 of 10 detections working, ANPR/face/rules/AI search are UI mocks only

---

## What Is the Demo Pack?

The Demo Pack is **not a product SKU**. It is a capability showcase that Techser uses to demo SafetyLens to implementation partners. The goal is wow factor: show 10+ detections, ANPR, face recognition, AI search, gate automation, and a rules engine all running on affordable hardware. Partners see this and decide to resell SafetyLens.

The demo runs on test `.mp4` videos (not live RTSP), on a desktop RTX 3050 PC or a MacBook with MPS. The inference pipeline is identical to production — only the video source differs.

```
Demo: test-videos/*.mp4 --> video_processor thread --> YOLO --> alerts
Prod: RTSP cameras      --> video_processor thread --> YOLO --> alerts
```

---

## Detection Features (10 total, 6 working)

| # | Detection | Model | Day 1 Accuracy | Status |
|---|-----------|-------|----------------|--------|
| 1 | Number Plate Recognition (ANPR) | YOLO26s + PaddleOCR | ~85-88% | **NOT BUILT** — pipeline not implemented |
| 2 | Face Detection + Recognition | InsightFace SCRFD + ArcFace | ~90-92% frontal | **NOT BUILT** — pipeline not implemented |
| 3 | Fall / Man Down | YOLO26n-pose | ~75-80% | **NOT BUILT** — pose model not integrated |
| 4 | Fire & Smoke | YOLOE zero-shot | ~65-75% fire, ~55-65% smoke | **DONE** — YOLOE text prompt |
| 5 | PPE — Helmet | YOLOE zero-shot | ~70-80% | **DONE** — YOLOE text prompt |
| 6 | PPE — Safety Vest | YOLOE zero-shot | ~70-78% | **DONE** — YOLOE text prompt |
| 7 | Person Counting & Zone Intrusion | YOLO26n COCO | ~93-95% | **DONE** — working in inference loop |
| 8 | Vehicle Classification | YOLO26n COCO | ~88-90% | **DONE** — COCO classes (car, truck, bus, motorcycle) |
| 9 | Mobile Phone Usage | YOLOE zero-shot | ~60-70% | **DONE** — YOLOE or COCO class 67 |
| 10 | Smoking Detection | YOLOE zero-shot | ~55-65% | **PARTIAL** — YOLOE can detect, not specifically tested |

---

## Modules — Detailed Status

### ANPR Module (NOT BUILT)

| Feature | Description | Status |
|---------|-------------|--------|
| Plate detection | YOLO26s detects plate bounding box | **NOT BUILT** — model not trained/integrated |
| Plate OCR | PaddleOCR reads text from plate crop | **NOT BUILT** — PaddleOCR not in requirements.txt |
| Vehicle log | Plate + timestamp + camera logged | **NOT BUILT** — no `plate_reads` table |
| Plate database | Whitelist / blacklist / visitor lists | **UI MOCK ONLY** — `PlateManagement.tsx` with mock data |
| Plate search | Search by partial plate number | **UI MOCK ONLY** |

**What needs to happen:**
1. Train or fine-tune YOLO26s for Indian number plates
2. Add PaddleOCR to `requirements.txt` and integrate into inference loop
3. Create `plate_reads` + `plate_lists` tables in PostgreSQL
4. Build CRUD API endpoints for plate management
5. Connect `PlateManagement.tsx` to real backend (replace mock data)

**Effort estimate:** ~5 days

---

### Face Recognition Module (NOT BUILT)

| Feature | Description | Status |
|---------|-------------|--------|
| Face detection | SCRFD-2.5GF detects faces | **NOT BUILT** — InsightFace not in requirements.txt |
| Face embedding | ArcFace 512-d vector per face | **NOT BUILT** |
| Face enrollment | Photo upload or live capture | **UI MOCK ONLY** — `FaceEnrollment.tsx` with mock data |
| Face matching | Compare against enrolled DB | **NOT BUILT** — no `enrolled_faces` table |
| Face log | Match/non-match logged | **NOT BUILT** — no `face_logs` table |
| DPDPA consent | Consent checkbox + method dropdown | **UI MOCK ONLY** — consent flow in enrollment modal |

**What needs to happen:**
1. Add InsightFace (SCRFD + ArcFace) to `requirements.txt`
2. Install pgvector extension for PostgreSQL (embedding storage)
3. Create `enrolled_faces` + `face_logs` tables
4. Build enrollment API (photo upload, embedding extraction, DB insert)
5. Add face matching step in inference loop (compare detections against enrolled DB)
6. Connect `FaceEnrollment.tsx` to real backend (replace mock data)

**Effort estimate:** ~5 days

---

### AI Search Module (NOT BUILT — Phase 2)

| Feature | Description | Status |
|---------|-------------|--------|
| Embedding pipeline | MobileCLIP encodes detection crops into pgvector | **NOT BUILT** |
| Text search | CLIP text encoder matched via cosine similarity | **NOT BUILT** |
| Search UI | Filter-first search with thumbnail grid | **UI MOCK ONLY** — `AISearch.tsx` with mock data |
| Re-ID | Find similar person across cameras | **NOT BUILT** |

**What needs to happen:**
1. Load MobileCLIP model (`mobileclip_blt.ts` exists at 455 MB in repo root)
2. Install and configure pgvector extension
3. Create `detection_embeddings` table
4. Build background embedding thread (encode detection crops after each frame)
5. Build search API endpoint (text query -> CLIP text encoding -> cosine similarity)
6. Connect `AISearch.tsx` to real backend (replace mock data)

**Effort estimate:** ~5 days

---

### Rules Engine (PARTIAL)

| Feature | Description | Status |
|---------|-------------|--------|
| Rule cards UI | List of WHEN/IF/THEN/ELSE rules | **UI MOCK ONLY** — `RulesEngine.tsx` with mock data |
| Rule editor UI | Full-page create/edit with conditions + actions | **UI MOCK ONLY** — `RuleEditor.tsx` |
| Rule evaluation backend | Evaluate rules after each detection | **NOT BUILT** — no `evaluate_rules()` function |
| Rules database | Rules stored in PostgreSQL | **NOT BUILT** — no `rules` table |
| Gate automation | ANPR + face result triggers open/close gate via HTTP/relay | **NOT BUILT** |
| PLC/webhook actions | Trigger external systems from rules | **NOT BUILT** |
| Preset templates | PPE, Fire, Gate Entry, After-Hours, Overcrowding | **UI MOCK ONLY** — presets in `mockRules.ts` |

**What needs to happen:**
1. Create `rules` table in PostgreSQL
2. Build CRUD API endpoints for rule management
3. Implement `evaluate_rules()` — runs after each detection in the inference loop
4. Build action executors (`create_alert` already works; need `open_gate`, `webhook`, `plc_trigger`, etc.)
5. Connect `RulesEngine.tsx` and `RuleEditor.tsx` to real backend (replace mock data)

**Effort estimate:** ~5 days

---

### License System (NOT BUILT)

| Feature | Description | Status |
|---------|-------------|--------|
| License status page | Shows customer, cameras, features, expiry | **UI MOCK ONLY** — `LicenseStatus.tsx` with hardcoded data |
| Signed `.lic` file | Ed25519 signature verification | **NOT BUILT** |
| Feature gating | Hide/disable unlicensed features | **NOT BUILT** |
| License hub | Admin dashboard to generate licenses (separate project) | **NOT BUILT** |
| Dev toggle | Shift+L x5 to cycle license states | **DONE** — in `LicenseStatus.tsx` |

**Not needed for partner demo.** License validation matters for production deployments, not the demo.

---

### Camera Roles (UI ONLY)

| Feature | Description | Status |
|---------|-------------|--------|
| Role dropdown | General / Gate (ANPR+Face) / Work Zone / Manual | **UI MOCK ONLY** — added to `CameraConfig.tsx` |
| Model routing | Load different models based on camera role | **NOT BUILT** — all cameras run same models |
| GPU impact display | Show estimated GPU cost per role | **UI MOCK ONLY** — hardcoded estimates |

**Context:** Camera roles are how the demo explains model routing. A "Gate" camera would run ANPR + face models, a "Work Zone" camera would run PPE + phone models. This is important for GPU budget management on real deployments. For the demo, the concept is shown in the UI but the backend treats all cameras identically.

---

## What IS Built and Working (Demo Ready)

These features work end-to-end with real inference, real alerts, real Telegram notifications:

| # | Feature | Key Files |
|---|---------|-----------|
| 1 | Multi-camera live view with detection overlays (MJPEG + WebSocket) | `backend/server.py`, `frontend/src/pages/LiveView.tsx` |
| 2 | 6 of 10 detections (person, zone intrusion, phone, fire/smoke, PPE, vehicle) | `backend/server.py`, `yolo26n.pt`, `yoloe-11s-seg.pt` |
| 3 | Alert system — PostgreSQL persistence, cooldown, severity, acknowledge/resolve/snooze | `backend/alert_store.py`, `frontend/src/pages/AlertCenter.tsx` |
| 4 | Telegram alerts — snapshots with bounding boxes, <3s latency | `backend/telegram_notifier.py` |
| 5 | Zone management — draw polygons, enforce in inference loop | `frontend/src/pages/ZoneManagement.tsx`, `backend/server.py` |
| 6 | Dashboard — KPIs, time-series charts, violation breakdowns | `frontend/src/pages/Dashboard.tsx` |
| 7 | Camera config — CRUD, demo mode selection (yolo/yoloe/yolo+vlm) | `frontend/src/pages/CameraConfig.tsx` |
| 8 | System settings — live config changes (FPS, confidence, cooldown) | `frontend/src/pages/SystemSettings.tsx`, `backend/config_manager.py` |
| 9 | Docker deployment — single `docker compose up` with GPU passthrough | `Dockerfile`, `docker-compose.yml` |

These features have UI that renders with mock data but **no backend**:

| # | Feature | Key Files |
|---|---------|-----------|
| 10 | Rules Engine — WHEN/IF/THEN rule cards and editor | `RulesEngine.tsx`, `RuleEditor.tsx`, `mockRules.ts` |
| 11 | Vehicle Plates (ANPR) — plate lists, recent reads | `PlateManagement.tsx`, `mockPlates.ts` |
| 12 | Face Enrollment — face cards, enrollment modal, DPDPA consent | `FaceEnrollment.tsx`, `mockFaces.ts` |
| 13 | AI Search — filter-first search with thumbnail grid | `AISearch.tsx`, `mock.ts` |
| 14 | License Status — status card, features list, upload area | `LicenseStatus.tsx` |
| 15 | Camera Roles — role dropdown with GPU impact display | `CameraConfig.tsx` |

---

## Models

| Model | File | Size | In Repo? | Purpose |
|-------|------|------|----------|---------|
| YOLO26n | `yolo26n.pt` | 5.3 MB | Exists (not in git) | COCO 80-class detection |
| YOLOE-11s-seg | `yoloe-11s-seg.pt` | 27 MB | Exists (not in git) | Zero-shot PPE/fire/phone via text prompts |
| YOLO26s | -- | ~15 MB | **MISSING** | Plate detection (ANPR) |
| PaddleOCR | -- | ~10 MB | **MISSING** | Plate text reading |
| SCRFD-2.5GF | -- | ~5 MB | **MISSING** | Face detection |
| ArcFace | -- | ~170 MB | **MISSING** | Face embedding/matching |
| MobileCLIP | `mobileclip_blt.ts` | 455 MB | Exists (not in git) | AI Search embeddings |
| YOLO26n-pose | -- | ~6 MB | **MISSING** | Fall detection |

Model weights are **not** in the git repo. Download separately or copy from model storage.

---

## Hardware

### RTX 3050 6GB Desktop (~INR 45,000)

- i5-12400F, Amazon Basics H610, 16GB DDR4, 500GB NVMe
- Handles 10 cameras at ~45% GPU with current models
- With ANPR + face on 3 gate cameras: ~45% GPU still (smart model routing)
- This is the primary demo hardware

### MacBook with MPS

- Works for local development and small demos
- MPS backend for PyTorch inference

### Jetson Orin NX 16GB (~INR 1,00,000)

- Production deployment target
- Requires TensorRT FP16 export (not built)
- Not needed for the demo

---

## File Structure (Key Files)

```
video-analytics/
├── backend/
│   ├── server.py              — FastAPI app, inference engine, all API endpoints
│   ├── alert_store.py         — PostgreSQL alert CRUD
│   ├── config_manager.py      — Thread-safe config with atomic writes
│   ├── telegram_notifier.py   — Telegram Bot API integration
│   ├── config.example.json    — Config template (copy to config.json)
│   └── tests/                 — Unit tests
├── frontend/
│   ├── src/pages/
│   │   ├── LiveView.tsx       — Multi-camera MJPEG grid
│   │   ├── AlertCenter.tsx    — Alert feed with filters
│   │   ├── Dashboard.tsx      — KPIs + charts
│   │   ├── CameraConfig.tsx   — Camera CRUD + roles (role dropdown is mock)
│   │   ├── RulesEngine.tsx    — Rule cards (mock data)
│   │   ├── RuleEditor.tsx     — Full-page rule editor (mock data)
│   │   ├── PlateManagement.tsx — ANPR plates (mock data)
│   │   ├── FaceEnrollment.tsx — Face enrollment (mock data)
│   │   ├── AISearch.tsx       — AI search (mock data)
│   │   ├── LicenseStatus.tsx  — License page (mock data)
│   │   └── ...
│   ├── src/data/
│   │   ├── mock.ts            — Shared mock data + types
│   │   ├── mockRules.ts       — Rules engine mock data
│   │   ├── mockPlates.ts      — ANPR mock data
│   │   └── mockFaces.ts       — Face enrollment mock data
│   └── src/lib/api.ts         — API client (all endpoints)
├── Dockerfile                 — Multi-stage (Node build + CUDA runtime)
├── docker-compose.yml         — Postgres + backend with GPU
├── requirements.txt           — Python dependencies
└── docs/                      — PRD, UI guides, specs
```

---

## What to Build Next

### Must Have for Partner Demo

These are the features that deliver the wow factor. Without them, the demo looks like a basic CCTV overlay.

**1. ANPR pipeline**
- YOLO26s plate detection + PaddleOCR text reading
- Log plate reads to PostgreSQL, connect to `PlateManagement.tsx`
- Dependencies: YOLO26s model (train/fine-tune for Indian plates), PaddleOCR package
- Effort: ~5 days

**2. Face recognition pipeline**
- SCRFD face detection + ArcFace embedding + matching against enrolled DB
- Log face events, connect to `FaceEnrollment.tsx`
- Dependencies: InsightFace package, pgvector extension
- Effort: ~5 days

**3. Rules engine backend**
- `evaluate_rules()` running after each detection, `rules` table, CRUD API
- Connect to `RulesEngine.tsx` and `RuleEditor.tsx`
- Dependencies: None (PostgreSQL already running)
- Effort: ~5 days

### Nice to Have for Demo

**4. AI Search backend**
- MobileCLIP embedding pipeline + pgvector cosine similarity search
- Connect to `AISearch.tsx`
- Dependencies: MobileCLIP model (exists in repo), pgvector
- Effort: ~5 days

**5. Fall detection**
- YOLO26n-pose integration into inference loop
- Dependencies: Pose model download
- Effort: ~2 days

**6. Gate automation demo**
- Simulated boom barrier (HTTP endpoint) triggered by rules engine
- Dependencies: Rules engine backend (item 3)
- Effort: ~1 day

### Not Built Anywhere (no UI, no backend)

These have zero implementation — not even mock UI:

| Feature | Why it matters | When to build |
|---------|---------------|---------------|
| **User management / RBAC** | No login, no roles, no permissions. Anyone with network access can change config, delete alerts, modify rules. Operators should not edit rules; integrators should not acknowledge alerts during setup. | Before first production deployment. Not needed for demo. |
| **Feature gating (greyed-out disabled states)** | Unlicensed features should show as disabled with "Contact vendor to upgrade" message. Currently all features are always visible and functional. No visual distinction between licensed and unlicensed. | Alongside license validation. Not needed for demo. |
| **Audit log** | No record of who changed what — who acknowledged an alert, who modified a rule, who enrolled a face. Required for ISO 45001 safety audits. | Before production. Not needed for demo. |
| **System health dashboard** | No visibility into CPU/GPU/RAM usage, camera connection status, inference FPS, disk usage. Operators have no idea if the system is degraded. | Before production. Nice-to-have for demo. |

### Not Needed for Demo

These are production concerns, not demo concerns:

- TensorRT export (demo runs on desktop, not Jetson)
- License validation (demo does not need licensing)
- Email / WhatsApp alerts (Telegram is enough)
- PDF / CSV reports (dashboard is enough)

---

## Repository

- Code: https://github.com/lazycoder1/safety-lens-software
- Branch: `master`
- Sensitive docs excluded (pricing, competitive analysis, client requirements are in `/workspace/techser/docs-internal/`)
