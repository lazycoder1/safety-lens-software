# SafetyLens — Product Requirements Document (PRD)

**Version:** 2.0
**Date:** 2026-03-28
**Author:** Gautham G Sabhahit
**Target Customer:** Industrial factories, warehouses, logistics hubs
**First Customer:** TMEIC

---

## 1. Product Vision

SafetyLens is an on-premise, AI-powered video analytics platform for industrial safety and security. It processes live camera feeds to detect safety violations, recognize vehicles and faces, and trigger automated responses — all running on a single edge device with no cloud dependency.

**Two layers:**
1. **Base Platform** — general-purpose detections, alert engine, rules engine, dashboard. Works for any factory.
2. **Add-on Modules** — ANPR, face recognition, AI search, PLC integrations. Activated per deployment.

**Deployment model: 100% on-premise.** No video data leaves the premises. Licensing is via a signed license file — no ongoing internet required.

---

## 2. Architecture Overview

```
IP Cameras (RTSP)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Edge Device (RTX 3050 PC / Jetson Orin NX)                  │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Decode Layer │  │ Model Router │  │ Detection Models   │  │
│  │ RTSP → frame │──│ routes frame │──│ YOLO26n (COCO)     │  │
│  │ 1080p, 5FPS  │  │ by camera    │  │ YOLOE-11s (zero)   │  │
│  └─────────────┘  │ role + zone  │  │ YOLO26s (ANPR)     │  │
│                    └──────────────┘  │ SCRFD (face det)   │  │
│                                      │ ArcFace (face emb) │  │
│                                      │ PaddleOCR (plate)  │  │
│                                      │ MobileCLIP (search)│  │
│                                      └────────────────────┘  │
│                                              │                │
│                                              ▼                │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Rules Engine                                           │   │
│  │ WHEN: detection event                                  │   │
│  │ IF:   conditions (whitelist, zone, time, confidence)   │   │
│  │ THEN: actions (alert, gate, telegram, webhook, log)    │   │
│  └───────────────────────────────────────────────────────┘   │
│                    │                │                │        │
│                    ▼                ▼                ▼        │
│              ┌──────────┐   ┌──────────┐   ┌────────────┐   │
│              │ Alerts DB │   │ Gate API │   │ Telegram   │   │
│              │ (Postgres)│   │ Relay/PLC│   │ Webhook    │   │
│              └──────────┘   └──────────┘   └────────────┘   │
│                    │                                         │
│                    ▼                                         │
│              ┌──────────────────────────────────────┐       │
│              │ Web Dashboard (React, served locally) │       │
│              │ Live view, alerts, search, config     │       │
│              └──────────────────────────────────────┘       │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Embedding Pipeline (background)                        │  │
│  │ Detection crops → MobileCLIP → pgvector (AI Search)    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  License: signed .lic file, verified on startup (offline)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Detection Features

### 3.1 Base Detections (ship with every deployment)

| ID | Detection | Model | Day 1 Accuracy | Notes |
|----|-----------|-------|----------------|-------|
| D-1 | Person Detection | YOLO26n (COCO) | ~93-95% | Foundation for counting, zone intrusion |
| D-2 | Vehicle Classification | YOLO26n (COCO) | ~88-90% | Car, truck, bus, motorcycle, bicycle |
| D-3 | PPE — Helmet | YOLOE zero-shot | ~70-80% | Improvable to ~85-90% with fine-tuning |
| D-4 | PPE — Safety Vest | YOLOE zero-shot | ~70-78% | Improvable to ~85-88% with fine-tuning |
| D-5 | Fire & Smoke | YOLOE zero-shot | ~65-75% fire, ~55-65% smoke | Improvable to ~80-85% with fine-tuning |
| D-6 | Fall / Man Down | YOLO26n-pose | ~75-80% | Pose-based, vertical→horizontal transition |
| D-7 | Mobile Phone Usage | YOLOE zero-shot | ~60-70% | Improvable to ~80-85% with fine-tuning |
| D-8 | Smoking | YOLOE zero-shot | ~55-65% | Improvable to ~75-80% with fine-tuning |
| D-9 | Zone Intrusion | YOLO26n + polygon ROI | ~93-95% | Person in restricted area |
| D-10 | Person Counting | YOLO26n (COCO) | ~93-95% | Per-zone headcount |

Zero-shot detections (D-3 to D-8) use YOLOE — new detections can be added by typing a keyword, no retraining. Fine-tuning improves accuracy by 10-20% but is not included in the base package (estimated 10-20 man-hours per detection using public datasets, 80-100 hours total).

### 3.2 ANPR Module

| ID | Feature | Description | Status |
|----|---------|-------------|--------|
| ANPR-1 | Plate detection | YOLO26s detects plate bounding box in frame | Not built |
| ANPR-2 | Plate OCR | PaddleOCR reads text from plate crop | Not built |
| ANPR-3 | Vehicle log | Every plate read logged with timestamp, camera, direction, snapshot | Not built |
| ANPR-4 | Plate database | Whitelist / blacklist / visitor lists, editable from dashboard | Not built |
| ANPR-5 | Plate search | Search historical reads by full or partial plate number | Not built |

Expected accuracy: ~85-88% day 1, ~92-94% with fine-tuning on local plate styles.
Runs on gate cameras only (2-3 cameras), not all 10.

### 3.3 Face Recognition Module

| ID | Feature | Description | Status |
|----|---------|-------------|--------|
| FR-1 | Face detection | SCRFD-2.5GF detects faces in frame | Not built |
| FR-2 | Face embedding | ArcFace generates 512-d embedding per face crop | Not built |
| FR-3 | Face enrollment | Enroll employees/visitors via photo upload or live capture | Not built |
| FR-4 | Face matching | Compare detected face against enrolled database, return match + confidence | Not built |
| FR-5 | Face log | Every match/non-match logged with timestamp, camera, person name, snapshot | Not built |
| FR-6 | Unknown face alert | Alert when unrecognized face detected at a gate camera | Not built |

Expected accuracy: ~90-92% frontal day 1, ~95%+ with SCRFD-10GF upgrade.
Runs on gate cameras only. Requires consent workflow (India DPDPA compliance).

### 3.4 AI Search Module (Phase 2)

| ID | Feature | Description | Status |
|----|---------|-------------|--------|
| AS-1 | Embedding pipeline | Crop each detection, encode with MobileCLIP, store vector in pgvector | Not built |
| AS-2 | Text search | User types description ("red shirt", "forklift") → CLIP text encoding → cosine similarity search | Not built |
| AS-3 | Search results UI | Thumbnail grid with timestamps, camera, confidence — click to view context | Not built |
| AS-4 | Filters | Filter by camera, time range, detection class | Not built |
| AS-5 | Re-ID (stretch) | "Find this person across all cameras" — click a detection, find similar embeddings | Not built |

Fully offline — no LLM or API needed. MobileCLIP (~200MB) runs on the same GPU.
AI search is an add-on that layers on top of core detections — detections must be running first.

---

## 4. Rules Engine

The rules engine is the core of SafetyLens automation. Every alert, gate action, and notification flows through it.

### 4.1 Concept

```
WHEN:  <trigger>        — a detection event occurs
IF:    <conditions>      — optional filters (zone, time, whitelist, confidence)
THEN:  <actions>         — what to do (alert, open gate, send telegram, webhook)
ELSE:  <fallback actions> — what to do if conditions fail
```

### 4.2 Triggers

| Trigger | Fires when |
|---------|-----------|
| `detection` | Any YOLO/YOLOE detection above confidence threshold |
| `plate_read` | ANPR reads a plate |
| `face_match` | Face recognized (match found in enrolled DB) |
| `face_unknown` | Face detected but no match |
| `zone_enter` | Person/vehicle enters a zone |
| `zone_exit` | Person/vehicle leaves a zone |
| `count_threshold` | Person count in zone exceeds limit |

### 4.3 Conditions

| Condition | Example |
|-----------|---------|
| `zone` | Only in "Gate 1" or "Assembly Line" |
| `time_window` | Only during 09:00-18:00 |
| `day_of_week` | Only on weekdays |
| `plate_in_list` | Plate is in whitelist / blacklist / visitor list |
| `face_in_group` | Face is in "employees" / "visitors" / "contractors" group |
| `confidence_above` | Detection confidence > 0.8 |
| `class_is` | Detection class is "fire" or "no_helmet" |
| `count_exceeds` | Person count > N |

### 4.4 Actions

| Action | What it does |
|--------|-------------|
| `create_alert` | Create alert in DB with severity (P1-P4) |
| `send_telegram` | Send snapshot + caption to Telegram group |
| `send_webhook` | POST JSON payload to external URL |
| `open_gate` | Send HTTP/relay command to boom barrier controller |
| `close_gate` | Send close command |
| `log_entry` | Log attendance/vehicle entry with timestamp |
| `play_sound` | Trigger audio alert on dashboard |
| `trigger_plc` | Send command to PLC (siren, strobe, speaker) |
| `send_email` | Send alert email with snapshot |

### 4.5 Example Rules

**Rule: Auto-open gate for whitelisted vehicles**
```
WHEN:   plate_read on camera "Gate 1 Entry"
IF:     plate_in_list "whitelist" AND face_in_group "employees"
THEN:   open_gate "boom_barrier_1", log_entry
ELSE IF: plate_in_list "whitelist" AND face_unknown
THEN:   create_alert P2 "Known vehicle, unknown driver", send_telegram
ELSE:   create_alert P1 "Unknown vehicle", close_gate, send_telegram
```

**Rule: PPE violation in work zone**
```
WHEN:   detection class "no_helmet" on cameras tagged "work_zone"
IF:     zone is "Assembly Line" AND confidence_above 0.7
THEN:   create_alert P2, send_telegram
```

**Rule: Fire — immediate response**
```
WHEN:   detection class "fire" OR "smoke"
IF:     confidence_above 0.6
THEN:   create_alert P1, send_telegram, trigger_plc "siren", send_webhook
```

**Rule: After-hours intrusion**
```
WHEN:   zone_enter "Warehouse"
IF:     time_window 22:00-06:00
THEN:   create_alert P1, send_telegram, play_sound
```

### 4.6 Data Model

```sql
rules (
    id          UUID PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    enabled     BOOLEAN DEFAULT true,
    priority    INTEGER DEFAULT 0,        -- higher = evaluated first
    trigger     TEXT NOT NULL,             -- detection, plate_read, face_match, etc.
    cameras     TEXT[],                    -- NULL = all cameras
    conditions  JSONB NOT NULL DEFAULT '[]',
    actions     JSONB NOT NULL,
    else_actions JSONB,
    cooldown_seconds INTEGER DEFAULT 60,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
)

-- Example conditions JSONB:
-- [
--   {"type": "plate_in_list", "list": "whitelist"},
--   {"type": "time_window", "start": "09:00", "end": "18:00"},
--   {"type": "confidence_above", "value": 0.7}
-- ]

-- Example actions JSONB:
-- [
--   {"type": "create_alert", "severity": "P2"},
--   {"type": "send_telegram"},
--   {"type": "open_gate", "device": "boom_barrier_1"}
-- ]
```

### 4.7 Backend Evaluation

```python
# Pseudocode — runs after every detection
def evaluate_rules(event):
    rules = get_enabled_rules(trigger=event.type, camera=event.camera_id)
    for rule in sorted(rules, key=lambda r: -r.priority):
        if is_cooling_down(rule, event):
            continue
        if all_conditions_met(rule.conditions, event):
            execute_actions(rule.actions, event)
            break  # first matching rule wins (priority order)
        elif rule.else_actions:
            execute_actions(rule.else_actions, event)
            break
```

---

## 5. Smart Model Routing

Not all models run on all cameras. Each camera has a role that determines which models process its frames.

| Camera Role | Models | GPU cost/frame (RTX 3050) |
|-------------|--------|---------------------------|
| General surveillance | YOLO26n + YOLOE-11s | ~5ms |
| Gate (ANPR + face) | YOLO26n + YOLOE-11s + YOLO26s + PaddleOCR + SCRFD + ArcFace | ~18ms |
| Work zone (PPE focus) | YOLO26n + YOLOE-11s | ~5ms |

With 7 general + 3 gate cameras at 5 FPS: ~45% GPU utilization on RTX 3050 6GB.

---

## 6. Camera & Streaming

| Setting | Value | Rationale |
|---------|-------|-----------|
| Source resolution | 1080p | YOLO resizes all input to 640x640 internally. 4K gives zero accuracy improvement, only 4x more decode load (800 Mbps vs 200 Mbps). |
| Processing FPS | 5 per camera | 50 frames/sec total across 10 cameras. Sufficient for safety events (not speed cameras). |
| Stream protocol | RTSP (H.264/H.265) | Standard IP camera protocol |
| Dashboard stream | MJPEG via HTTP | Simple, works in any browser, no WebRTC complexity |

---

## 7. Alert System

### 7.1 Severity Levels

| Level | Color | Use | Response |
|-------|-------|-----|----------|
| P1 — Critical | Red | Fire, zone intrusion, fall, unknown person at gate | Immediate — audio alert, telegram, possible PLC trigger |
| P2 — High | Orange | No helmet, no vest, unknown vehicle | Urgent — telegram, dashboard toast |
| P3 — Medium | Yellow | Phone usage, smoking, minor violations | Monitor — dashboard only |
| P4 — Info | Blue | Person count, vehicle log, attendance | Log — no notification |

### 7.2 Notification Channels

| Channel | Status | Use |
|---------|--------|-----|
| Dashboard toast | Done | All severities |
| WebSocket push | Done | Real-time to open dashboards |
| Telegram | Done | P1/P2 with snapshot |
| Email | Not built | Shift reports, escalations |
| Webhook | Not built | ERP/SCADA/SAP integration |
| PLC / relay | Not built | Siren, strobe, boom barrier |

### 7.3 Alert Lifecycle

```
Active → Acknowledged → Resolved
  │
  └──→ Snoozed (with duration) → Active (when snooze expires)
  │
  └──→ False Positive (removed from stats)
```

---

## 8. Licensing

### 8.1 Signed License File (Offline)

No license server required at runtime. A signed `.lic` file is generated from the admin dashboard and transferred to the edge device via USB or email.

```json
{
  "license_id": "SL-2026-0001",
  "customer": "TMEIC Jamshedpur",
  "max_cameras": 10,
  "features": ["base", "anpr", "face", "ai_search"],
  "issued_at": "2026-03-26",
  "expires_at": "2027-03-26",
  "hardware_id": null,
  "signature": "<ed25519-signature>"
}
```

- Signed with Techser's private key (Ed25519)
- Verified with public key baked into SafetyLens Docker image
- Tamper-proof — change one byte and signature fails
- No internet needed, ever

### 8.2 License Hub (Techser admin tool)

Separate project — Next.js on Vercel + Neon Postgres.

- Generate and sign license files
- Track issued licenses, customers, expiry dates
- Renew / revoke (issue new file, old one expires naturally)
- Link to Freshdesk for support tickets

### 8.3 Edge Enforcement

| Check | When | Behavior |
|-------|------|----------|
| Signature valid | Every startup | Invalid → system refuses to start, shows activation screen |
| Not expired | Every startup | Expired → 7-day grace period (warns), then stops inference |
| Camera count | Camera add | Over limit → "Upgrade license" message |
| Feature check | Model load | Unlicensed feature → "Contact vendor to upgrade" in UI |
| Hardware binding (optional) | Startup | If set, SHA-256 of CPU+GPU+MAC must match |

---

## 9. Deployment

### 9.1 Docker Compose

Single command deployment: `docker compose up --build`

| Service | Image | Purpose |
|---------|-------|---------|
| `db` | postgres:16-alpine + pgvector | Alerts, rules, plates, faces, embeddings |
| `backend` | nvidia/cuda:12.4 + Python 3.11 | FastAPI + inference engine + serves frontend |

Frontend is built at Docker build time and served by FastAPI as static files.

### 9.2 Hardware Requirements

**Option A: RTX 3050 PC (~45k INR)**

| Part | Spec |
|------|------|
| CPU | Intel Core i5-12400F (6C/12T) |
| GPU | ZOTAC RTX 3050 6GB |
| Motherboard | Amazon Basics H610 (LGA 1700) |
| RAM | 16GB DDR4-3200 |
| SSD | 500GB NVMe |
| PSU | 450W 80+ Bronze |

CPU, GPU, and SSD are critical. Motherboard, RAM brand, PSU, and cabinet can be substituted based on local availability.

**Option B: Jetson Orin NX 16GB (~1L INR)**

Compact edge form factor. Requires DLA offloading and INT8 quantization to handle full 10-camera pipeline. Tight but workable.

### 9.3 Server Prerequisites

- Ubuntu 22.04 LTS
- Docker Engine + Docker Compose v2
- NVIDIA driver 535+
- NVIDIA Container Toolkit

---

## 10. Data Model (Postgres)

### 10.1 Core Tables

```sql
-- Alerts (existing, working)
alerts (
    id TEXT PRIMARY KEY,
    severity TEXT,              -- P1, P2, P3, P4
    status TEXT,                -- active, acknowledged, resolved, snoozed
    rule TEXT,
    camera_id TEXT,
    camera_name TEXT,
    zone TEXT,
    confidence DOUBLE PRECISION,
    timestamp TEXT,
    source TEXT,
    description TEXT,
    snapshot_path TEXT,
    acknowledged_by TEXT,
    acknowledged_at TEXT,
    resolved_at TEXT,
    snoozed_until TEXT,
    false_positive BOOLEAN
)

-- Rules engine (new)
rules (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    enabled BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    trigger TEXT NOT NULL,
    cameras TEXT[],
    conditions JSONB DEFAULT '[]',
    actions JSONB NOT NULL,
    else_actions JSONB,
    cooldown_seconds INTEGER DEFAULT 60,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)

-- ANPR (new)
plate_reads (
    id UUID PRIMARY KEY,
    plate_text TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    camera_id TEXT,
    timestamp TIMESTAMPTZ,
    direction TEXT,            -- entry, exit
    snapshot_path TEXT,
    vehicle_class TEXT,        -- car, truck, bus, motorcycle
    matched_list TEXT          -- whitelist, blacklist, visitor, NULL
)

plate_lists (
    id UUID PRIMARY KEY,
    list_name TEXT NOT NULL,   -- whitelist, blacklist, visitor
    plate_text TEXT NOT NULL,
    owner_name TEXT,
    vehicle_desc TEXT,
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ
)

-- Face recognition (new)
enrolled_faces (
    id UUID PRIMARY KEY,
    person_name TEXT NOT NULL,
    group_name TEXT,           -- employees, visitors, contractors, blacklist
    embedding VECTOR(512),     -- pgvector
    photo_path TEXT,
    enrolled_at TIMESTAMPTZ,
    valid_until TIMESTAMPTZ
)

face_logs (
    id UUID PRIMARY KEY,
    camera_id TEXT,
    timestamp TIMESTAMPTZ,
    matched_face_id UUID REFERENCES enrolled_faces(id),
    confidence DOUBLE PRECISION,
    snapshot_path TEXT,
    is_unknown BOOLEAN DEFAULT false
)

-- AI Search (new, Phase 2)
detection_embeddings (
    id UUID PRIMARY KEY,
    camera_id TEXT,
    timestamp TIMESTAMPTZ,
    class_label TEXT,
    confidence DOUBLE PRECISION,
    bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER,
    thumbnail_path TEXT,
    embedding VECTOR(512)      -- MobileCLIP embedding, pgvector
)
```

---

## 11. API Endpoints (new/updated)

### 11.1 Rules Engine

```
GET    /api/rules                      — List all rules
POST   /api/rules                      — Create rule
PUT    /api/rules/{id}                 — Update rule
DELETE /api/rules/{id}                 — Delete rule
PUT    /api/rules/{id}/toggle          — Enable/disable rule
POST   /api/rules/test                 — Dry-run rule against sample event
```

### 11.2 ANPR

```
GET    /api/plates/reads               — Query plate reads (filter by plate, camera, time)
GET    /api/plates/lists               — Get all plate lists
POST   /api/plates/lists               — Add plate to list
PUT    /api/plates/lists/{id}          — Update plate entry
DELETE /api/plates/lists/{id}          — Remove plate from list
POST   /api/plates/lists/import        — Bulk import CSV
GET    /api/plates/search?q=KA05       — Search by partial plate
```

### 11.3 Face Recognition

```
GET    /api/faces                      — List enrolled faces
POST   /api/faces/enroll               — Enroll face (photo upload)
POST   /api/faces/enroll/live          — Enroll from live camera capture
DELETE /api/faces/{id}                 — Remove enrolled face
GET    /api/faces/logs                 — Query face match logs
GET    /api/faces/groups               — List groups (employees, visitors, etc.)
```

### 11.4 AI Search

```
POST   /api/search                     — Text query → matching detections
GET    /api/search/similar/{id}        — Find similar to a given detection
```

---

## 12. Build Priority

### Phase 1 — Core Platform (Month 1)

This phase delivers a working product with the 10 base detections.

| # | Task | Effort |
|---|------|--------|
| 1 | RTSP camera input (replace .mp4 files) | 3 days |
| 2 | Zone polygon ROI enforcement in inference loop | 3 days |
| 3 | Rules engine — backend (evaluate, persist, CRUD API) | 5 days |
| 4 | Rules engine — frontend (rule builder UI) | 5 days |
| 5 | Connect dashboard to real alert data (remove mock) | 2 days |
| 6 | Docker deployment + GPU passthrough tested on Linux | 3 days |
| 7 | License file validation on startup | 3 days |
| 8 | TensorRT FP16 export for all models | 2 days |

### Phase 2 — ANPR + Face Recognition (Month 2)

| # | Task | Effort |
|---|------|--------|
| 9 | ANPR pipeline (YOLO26s plate detect + PaddleOCR) | 5 days |
| 10 | Plate database (whitelist/blacklist CRUD + import) | 3 days |
| 11 | Face detection + embedding pipeline (SCRFD + ArcFace) | 5 days |
| 12 | Face enrollment UI (photo upload + live capture) | 3 days |
| 13 | Face matching + logging | 3 days |
| 14 | Gate automation rules (ANPR + face → open/close gate) | 3 days |
| 15 | Webhook / PLC action support in rules engine | 3 days |

### Phase 3 — AI Search + Polish (Month 3)

| # | Task | Effort |
|---|------|--------|
| 16 | MobileCLIP embedding pipeline (background thread) | 3 days |
| 17 | pgvector setup + storage | 2 days |
| 18 | Search API endpoint | 2 days |
| 19 | Search UI (text box, filters, thumbnail grid) | 5 days |
| 20 | Email alerts | 2 days |
| 21 | Shift reports + PDF export | 5 days |
| 22 | Fine-tuning pass on zero-shot detections | 10-15 days |
| 23 | On-site deployment + tuning | 5-10 days |

---

## 13. Dependencies & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| RTSP camera access at customer site | Blocks real testing | Test with RTSP simulator + recorded footage |
| Gate hardware specs (boom barrier API/relay) | Blocks gate automation | Get specs early, build generic HTTP/relay adapter |
| Face consent (DPDPA) | Legal risk | Enrollment requires explicit consent, data retention policy in UI |
| AI search storage growth | Disk fills up | Configurable retention (e.g., 30 days), auto-purge old embeddings |
| Zero-shot accuracy below customer expectations | Customer dissatisfaction | Set expectations upfront (Day 1 vs fine-tuned numbers), offer fine-tuning as paid add-on |

---

## 14. Out of Scope (This Release)

- Multi-site management / central dashboard
- Cloud deployment
- ERP/SAP deep integration (webhook only)
- Video recording / NVR functionality
- Re-ID across cameras (stretch goal in AI Search)
- Snake detection, drug detection (not feasible with standard cameras)

---

## 15. Success Criteria

| Metric | Target |
|--------|--------|
| Base detection accuracy (person, vehicle) | >= 90% |
| ANPR plate read accuracy | >= 85% day 1 |
| Face recognition accuracy (frontal, enrolled) | >= 90% |
| Alert latency (detection to dashboard) | < 2 seconds |
| False positive rate | < 10% after 1 week of tuning |
| Cameras supported | 10 on RTX 3050, 25+ on AGX Orin |
| Inference FPS per camera | >= 5 FPS |
| System uptime | 99% during working hours |
| Docker deployment time (fresh machine) | < 30 minutes |
