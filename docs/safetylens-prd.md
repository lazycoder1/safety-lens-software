# SafetyLens — Product Requirements Document (PRD)

**Version:** 1.0
**Date:** 2026-03-23
**Author:** Gautham G Sabhahit
**Target Customer:** TMEIC (first deployment), General industrial safety (product)

---

## 1. Product Vision

SafetyLens is an on-premise, AI-powered video analytics platform for industrial safety. It processes live camera feeds to detect safety violations and alerts in real-time.

The product is built in two layers:
1. **Base Platform** — general-purpose, works for any factory. Ships with standard detections.
2. **Configurable Features** — customer-specific detections and integrations, activated per deployment.

**Deployment model: 100% on-premise.** All processing, storage, and alerting runs on the customer's edge device. No video data leaves the premises. The only internet connection required is a one-time license activation call to the SafetyLens licensing server.

TMEIC is the first customer. The base platform must be solid enough to sell standalone; TMEIC-specific features are built as configurable add-ons.

---

## 2. Architecture Overview

```
IP Cameras (RTSP) ──► Edge Device (Jetson Orin NX / AGX Orin)
                          │
                          ├── Inference Engine (YOLO + TensorRT)
                          ├── Alert Engine (rules, cooldown, escalation)
                          ├── Zone Engine (polygon ROI, per-zone rules)
                          ├── Notification Layer (dashboard, Telegram, email)
                          ├── Training Pipeline (auto-label, retrain, deploy)
                          └── Web Dashboard (React, served via nginx)

Everything on-premise. No cloud dependency. No data leaves the device.
```

---

## 3. Feature Map

### 3.1 Base Platform (ships with every deployment)

These features are the product. Every customer gets them.

#### 3.1.1 Core Engine

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| BE-1 | RTSP camera input | Connect to IP cameras via RTSP URL | Add camera by URL, stream starts within 5s, auto-reconnect on drop | Not built (currently .mp4 only) |
| BE-2 | Multi-camera support (10) | Process up to 10 camera streams simultaneously | 10 streams at >= 5 FPS inference each on Orin NX | Partially built (3 cameras demo) |
| BE-3 | YOLO inference engine | Run trained YOLO model on every Nth frame | Configurable frame skip (1:1 to 1:10), TensorRT FP16 | Partially built (YOLOv8n on MPS, no TensorRT) |
| BE-4 | YOLOe open-vocab detection | Text-prompt based detection for any object | Add class by text, detection starts immediately, editable from UI | Done |
| BE-5 | Detection overlay | Draw bounding boxes + labels on camera feed | Boxes with class name, confidence %, color-coded by severity | Done |
| BE-6 | Config persistence | Camera, zone, and rule config survives restart | JSON/DB backed config, no data loss on reboot | Done (JSON) |

#### 3.1.2 Standard Detections (Base Pack — 4 detections)

| ID | Detection | TMEIC Ref | Method | Acceptance Criteria | Status |
|----|-----------|-----------|--------|---------------------|--------|
| SD-1 | Person / Zone Intrusion | #14, #15 | YOLO (person class) + polygon ROI | Alert when person enters marked restricted zone. < 2s latency. | Person detection working. Zone ROI not built. |
| SD-2 | Mobile Phone Usage | #10 | YOLO (cell phone class from COCO) | Detect phone in hand in work area. Flag with P3 alert. | Needs training on factory footage |
| SD-3 | Animal Intrusion (dog/cat) | #3 | YOLO (dog/cat class from COCO) | Detect stray animal on premises. Flag with P3 alert. | Not trained on factory footage |
| SD-4 | PPE — Helmet + Vest | #1 | YOLO (trained model) | Detect missing hard hat or high-vis vest. Flag with P2 alert. | Working (YOLOv8n trained) |

#### 3.1.3 Alert System

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| AL-1 | Real-time alert generation | Fire alert when violation detected | Alert within 2s of detection, includes camera name, rule, severity | Done (in-memory) |
| AL-2 | Cooldown / dedup | Don't fire same alert repeatedly | Configurable cooldown per rule (default 60s). No duplicate alerts within window. | Done |
| AL-3 | Alert persistence | Store alerts in database | SQLite DB, queryable by time/camera/rule/severity. Survives restart. | Not built (in-memory deque) |
| AL-4 | Alert acknowledge | Operator can acknowledge alert | Mark as acknowledged in UI, filter by ack status | Done (UI only, not persisted) |
| AL-5 | Dashboard alert delivery | Show alerts on web dashboard | Real-time WebSocket push, toast notification, alert feed with history | Done |
| AL-6 | Telegram alert delivery | Send violation snapshot to Telegram | Photo + caption (camera, rule, time) to configured Telegram group. Configurable per rule. | Not built |

#### 3.1.4 Zone Management

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| ZN-1 | Zone drawing UI | Draw polygon ROI on camera feed | Click-to-draw polygon, drag to adjust, save per camera. Min 3 points. | Not built |
| ZN-2 | Per-zone rules | Assign detection rules to specific zones | Each zone can have different rules (e.g., Zone A = helmet required, Zone B = no entry) | Not built |
| ZN-3 | Zone labels | Name and color-code zones | Display zone name overlay on camera feed | Not built |

#### 3.1.5 Web Dashboard

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| UI-1 | Live multi-camera view | Grid view of all cameras with overlays | 2x2, 3x3, 4x4 grid options. Click to maximize single camera. | Done |
| UI-2 | Alert feed | Scrollable list of alerts | Filter by severity, camera, rule, time range. Newest first. | Done |
| UI-3 | KPI dashboard | Violation counts, compliance % | Live data from alert DB. Today / this week / this month. | Done (mock data) |
| UI-4 | Camera management | Add/edit/delete cameras | CRUD for camera name, RTSP URL, assigned zones and rules | Done (for demo mode) |
| UI-5 | System settings | Global config | Frame skip rate, alert cooldown defaults, notification settings | Done |

#### 3.1.6 Deployment & Infrastructure

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| DP-1 | TensorRT export | Export YOLO model to TensorRT FP16 | Model loads and runs on Jetson/DGX with TensorRT. No PyTorch dependency at runtime. | Not built |
| DP-2 | Jetson Orin NX support | Run full stack on Orin NX | 10 cameras, 4 detections, dashboard — all on single Orin NX 16GB | Not tested |
| DP-3 | Auto-start on boot | System starts automatically | systemd service, auto-reconnect cameras, resume processing | Not built |
| DP-4 | Health monitoring | System health dashboard | CPU/GPU/RAM usage, camera connection status, inference FPS per camera | Not built |

#### 3.1.7 Product Licensing & Activation

The system must not run without a valid license. No one can deploy SafetyLens without explicit authorization.

**Activation Flow:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  FIRST BOOT (requires internet — one time only)                     │
│                                                                     │
│  1. Admin enters Product Key in setup wizard                        │
│     (provided by Jiffy Labs on purchase)                            │
│                                                                     │
│  2. Device collects hardware fingerprint:                           │
│     ├── CPU serial number                                           │
│     ├── GPU serial number (Jetson module ID)                        │
│     ├── MAC address (primary NIC)                                   │
│     └── Disk serial number                                          │
│                                                                     │
│  3. Device sends to SafetyLens License API:                         │
│     POST https://license.safetylens.ai/activate                     │
│     {                                                               │
│       "product_key": "SL-XXXX-XXXX-XXXX",                         │
│       "hardware_fingerprint": "<sha256 hash of hw identifiers>",   │
│       "requested_cameras": 10,                                      │
│       "requested_features": ["base", "ppe", "zone", ...]           │
│     }                                                               │
│                                                                     │
│  4. License server validates:                                       │
│     ├── Is this product key valid and unused?                       │
│     ├── Does the plan allow requested cameras/features?             │
│     └── Bind key to this hardware fingerprint (one device only)     │
│                                                                     │
│  5. Server returns signed license token:                            │
│     {                                                               │
│       "license_id": "...",                                          │
│       "customer": "TMEIC",                                         │
│       "max_cameras": 10,                                            │
│       "features": ["base", "ppe", "zone", "phone", "animal"],     │
│       "expires": "2027-03-23",                                      │
│       "hardware_hash": "<bound fingerprint>",                       │
│       "signature": "<RSA signed by Jiffy Labs private key>"        │
│     }                                                               │
│                                                                     │
│  6. Token stored locally as encrypted file                          │
│     License verified on every startup (offline, using public key)   │
│                                                                     │
│  AFTER ACTIVATION: No internet required. Fully air-gapped.          │
└─────────────────────────────────────────────────────────────────────┘
```

**License Enforcement:**

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| LIC-1 | Product key entry | Setup wizard requires product key on first boot | System does not start inference without a valid product key. Shows activation screen only. | Not built |
| LIC-2 | Hardware fingerprint | Bind license to specific device | Collect CPU serial, GPU serial, MAC, disk serial. SHA-256 hash. License cannot be moved to another device without reactivation. | Not built |
| LIC-3 | License API server | Cloud API to validate and issue licenses | Hosted at `license.safetylens.ai`. Validates product key, binds to hardware, returns signed token. Admin panel to manage keys. | Not built |
| LIC-4 | Signed license token | Cryptographically signed license file | RSA-2048 signed by Jiffy Labs private key. Public key embedded in SafetyLens binary. Cannot be forged or modified. | Not built |
| LIC-5 | Offline verification | Verify license on every startup without internet | Check token signature (RSA public key), verify hardware fingerprint matches current device, check expiry date. No network call needed. | Not built |
| LIC-6 | Camera count enforcement | Limit cameras to licensed count | If license allows 10 cameras, 11th camera add is rejected with "upgrade license" message. | Not built |
| LIC-7 | Feature gating | Enable/disable features per license | Each feature (detection module, integration) checks license before loading. Unlicensed features show "contact vendor to upgrade" in UI. | Not built |
| LIC-8 | License expiry | Annual license with renewal | 30-day warning before expiry. 7-day grace period after expiry (system warns but keeps running). After grace period, system stops inference, shows renewal screen. | Not built |
| LIC-9 | Tamper detection | Detect license file tampering | If license file is modified, deleted, or hardware changes, system stops and requires reactivation. Log tamper attempts. | Not built |
| LIC-10 | License admin panel | Web panel for Jiffy Labs to manage licenses | Generate product keys, view activations, revoke licenses, see active deployments, extend expiry. Hosted alongside license API. | Not built |

**Security Measures:**

| Layer | Method | What It Prevents |
|-------|--------|------------------|
| Product key | 20-char alphanumeric, single-use | Unauthorized installations |
| Hardware binding | SHA-256 of CPU + GPU + MAC + disk | Cloning the license to another device |
| RSA-2048 signature | Private key held by Jiffy Labs only | Forging or modifying license tokens |
| Encrypted storage | License file AES-256 encrypted on disk | Reading/copying the license file |
| Obfuscated binary | PyInstaller + code obfuscation (PyArmor) | Reverse engineering the license check |
| Heartbeat (optional) | Periodic phone-home if internet available | Detecting revoked licenses faster |

**How Secure Is This?**

| Threat | Protection | Difficulty to Break |
|--------|-----------|-------------------|
| Copy software to another machine | Hardware fingerprint binding — license won't validate on different hardware | Hard — requires matching CPU/GPU/MAC/disk serial |
| Forge a license file | RSA-2048 signature — requires Jiffy Labs private key | Very hard — RSA-2048 is industry standard |
| Modify license to add cameras/features | Signature check fails on any modification | Very hard |
| Bypass license check in code | PyArmor obfuscation + PyInstaller binary packaging | Medium — determined reverse engineer with weeks of effort could bypass, but this is standard for on-prem software |
| Share product key | Key is single-use and hardware-bound after first activation | Hard — key becomes useless after binding |

**Realistic assessment:** This is the same level of protection used by most commercial on-prem software (Milestone VMS, Genetec, etc.). No on-prem license system is 100% uncrackable — a determined attacker with physical access can eventually bypass any client-side check. But this raises the bar high enough that it's easier to just buy a license than to crack it. For enterprise/factory customers, this is more than sufficient.

**License API Requirements:**

| Component | Tech | Hosting | Cost |
|-----------|------|---------|------|
| License API | FastAPI or Flask | Any cheap VPS (DigitalOcean, Hetzner) or free tier (Railway, Render) | < $10/month |
| Database | PostgreSQL or SQLite | Same VPS | Included |
| Admin panel | Simple React or even Flask-Admin | Same VPS | Included |
| Domain | `license.safetylens.ai` | Cloudflare | ~$10/year |
| SSL | Let's Encrypt | Auto via Caddy/nginx | Free |

Total hosting cost for the licensing server: **< $15/month**. Handles thousands of activations.

---

### 3.2 Configurable Features (TMEIC Add-Ons)

These are built as modules that can be enabled per deployment. TMEIC needs all of them.

#### 3.2.1 Additional Detections — Phase 1 (Month 1, Committed)

| ID | Detection | TMEIC Ref | Method | Acceptance Criteria | Status |
|----|-----------|-----------|--------|---------------------|--------|
| CF-1 | Head cap vs helmet detection | #7 | YOLO (custom trained) | Distinguish production head cap from safety helmet in test/production areas. >= 85% accuracy. | Needs custom training data from TMEIC |
| CF-2 | Forklift operator without helmet | #9 | YOLO (forklift + person + helmet) | Detect forklift/pallet truck operator not wearing helmet. Requires association of person-on-vehicle with helmet status. | Needs forklift training data |
| CF-3 | Camera monitoring of test motor/drive | #13 | YOLO or YOLOe (custom) | Monitor test motor/drive via camera. Alert on anomaly (smoke, spark, unexpected state). Requires discussion with TMEIC on what "normal" vs "abnormal" looks like. | Needs requirements clarification |
| CF-4 | Risk zone marking around motor/equipment | #14 | YOLO + polygon ROI | Mark danger zones around motors/equipment. Alert when person enters. Depends on ZN-1 (zone drawing). | Blocked on ZN-1 |

#### 3.2.2 Additional Detections — Phase 2 (Month 2-3, Best-Effort)

| ID | Detection | TMEIC Ref | Method | Acceptance Criteria | Status |
|----|-----------|-----------|--------|---------------------|--------|
| CF-5 | Gangway / aisle blockage | #4 | VLM periodic scan | Detect objects blocking gangway/aisle. VLM analyzes scene every 30-60s. Alert if path obstructed. | VLM integration done, needs tuning |
| CF-6 | Safety belt — loading/unloading | #6 | YOLO (custom trained) | Detect missing safety belt while loading/unloading material. Requires specific camera angle and training data. >= 80% accuracy. | Needs training data |
| CF-7 | Safety belt — working at height | #8 | YOLO (custom trained) | Detect missing safety harness on boom lift or elevated work. Requires specific camera angle. >= 80% accuracy. | Needs training data |
| CF-8 | Fall detection + Fire detection | #11 | YOLO-pose (fall) + YOLO (fire/smoke) | Fall: detect person fallen using pose estimation. Fire: detect fire/smoke in camera view. Both P1 severity. | Needs pose model integration + fire/smoke training |
| CF-9 | Zone entry with auto-alert | #15 | YOLO + polygon ROI + PLC trigger | Person enters marked zone → trigger buzzer/strobe/speaker via PLC. Depends on ZN-1 + PLC integration. | Blocked on ZN-1 and PLC-1 |

#### 3.2.3 Features Requiring Further Discussion (TBD)

These face fundamental accuracy challenges. Not committed.

| ID | Detection | TMEIC Ref | Challenge | Next Step |
|----|-----------|-----------|-----------|-----------|
| TBD-1 | Plastic covers (snack covers) in hand | #2 | Small, context-dependent, partially hidden by hand. High false positive risk. | Need sample footage to evaluate feasibility |
| TBD-2 | Snake detection | #3 (partial) | Extremely small, camouflaged, rare event. Very difficult for YOLO. | Evaluate with VLM periodic scan |
| TBD-3 | Drugs/medicines/syringes | #5 | Extremely small objects. Contextual reasoning needed. Not feasible with current camera resolution. | Likely not feasible without close-up cameras |
| TBD-4 | Explosion / material fall | #11 (partial) | Rare, unpredictable. Very few training samples exist. | Evaluate fire/smoke as proxy |
| TBD-5 | Air gun + chemical usage without goggles | #12 | Two small objects that both need detection and correlation. | Need sample footage |

#### 3.2.4 Alerting — Phase 3

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| PLC-1 | PLC / IP Speaker integration | Trigger physical alarm on factory floor | Send signal to PLC on violation. Configurable per zone/rule. Buzzer/strobe/speaker. | Not built |
| NT-1 | Email alerts | Send violation email | Email with snapshot, camera name, rule, time. Configurable recipients per rule. | Not built |
| NT-2 | WhatsApp alerts | Send violation via WhatsApp | WhatsApp message with photo to configured number/group. | Not built |
| NT-3 | Webhook / API alerts | POST alert data to external system | JSON payload to configurable URL. For ERP/SCADA/SAP integration. | Not built |

---

### 3.3 Training Pipeline

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| TP-1 | Frame extraction from cameras | Auto-extract frames for training | Configurable: N frames per hour per camera. Saved to disk with metadata. | Script exists, not automated |
| TP-2 | Auto-labeling with YOLOe | Use YOLOe to pre-label extracted frames | YOLOe labels exported in YOLO format. Confidence threshold configurable. | Script exists |
| TP-3 | Roboflow integration | Upload labeled frames for human review | Push to Roboflow project, review in Roboflow UI, pull corrected labels. | Not built |
| TP-4 | Model retraining | Retrain YOLO on corrected labels | One-click retrain from UI or CLI. Outputs new .pt file with metrics. | Script exists, not in UI |
| TP-5 | Model hot-swap | Deploy new model without restart | Load new model weights, swap in inference engine, no downtime. | Not built |

---

### 3.4 Reporting

| ID | Feature | Description | Acceptance Criteria | Status |
|----|---------|-------------|---------------------|--------|
| RP-1 | Violation trend chart | 24h rolling violation count by type | Line chart, updates in real-time, filterable by camera/zone | Done (mock data) |
| RP-2 | Compliance heatmap | Zone x shift compliance matrix | Color-coded grid showing compliance % per zone per shift | Done (mock data) |
| RP-3 | Shift report | Auto-generated summary per shift | Total violations, top offending zones, comparison with previous shift | Not built |
| RP-4 | PDF/CSV export | Export reports for audits | Compliance report as PDF. Raw alert data as CSV. | Not built |

---

## 4. Build Plan — Priority Order

### P0 — Required for First Customer Demo (Week 1-2)

| # | Task | Feature IDs | Effort |
|---|------|-------------|--------|
| 1 | RTSP stream support | BE-1 | 2 days |
| 2 | Alert DB persistence (SQLite) | AL-3 | 2 days |
| 3 | Train YOLO on TMEIC footage | SD-1 to SD-4, CF-1 to CF-4 | 1 week (depends on footage availability) |
| 4 | TensorRT export + Jetson test | DP-1, DP-2 | 3 days |

### P1 — Base Platform Complete (Week 3-6)

| # | Task | Feature IDs | Effort |
|---|------|-------------|--------|
| 5 | Zone drawing UI | ZN-1, ZN-2, ZN-3 | 1 week |
| 6 | Per-zone rule assignment | ZN-2 | 3 days |
| 7 | Telegram alert integration | AL-6 | 2 days |
| 8 | Connect dashboard to real data | UI-3 | 3 days |
| 9 | Auto-start + health monitoring | DP-3, DP-4 | 2 days |
| 10 | Camera management for RTSP | UI-4 | 2 days |
| 11 | Product licensing — device side | LIC-1, LIC-2, LIC-4, LIC-5, LIC-6, LIC-7, LIC-8, LIC-9 | 1 week |
| 12 | License API server + admin panel | LIC-3, LIC-10 | 1 week |

### P2 — TMEIC Phase 1 Features (Week 4-8)

| # | Task | Feature IDs | Effort |
|---|------|-------------|--------|
| 13 | Head cap vs helmet model | CF-1 | 1 week (training) |
| 14 | Forklift operator detection | CF-2 | 1 week (training) |
| 15 | Motor/drive monitoring | CF-3 | 1 week (requires TMEIC input) |
| 16 | Risk zone alert (depends on ZN-1) | CF-4 | 2 days |

### P3 — TMEIC Phase 2 Features (Week 6-12)

| # | Task | Feature IDs | Effort |
|---|------|-------------|--------|
| 17 | Gangway blockage (VLM tuning) | CF-5 | 3 days |
| 18 | Safety belt detection models | CF-6, CF-7 | 2 weeks (training) |
| 19 | Fall detection (pose model) | CF-8 | 1 week |
| 20 | Fire/smoke detection | CF-8 | 1 week (training on public datasets) |
| 21 | PLC / buzzer integration | PLC-1, CF-9 | 1 week |
| 22 | Email + WhatsApp alerts | NT-1, NT-2 | 3 days |

### P4 — Polish & Scale (Week 10-14)

| # | Task | Feature IDs | Effort |
|---|------|-------------|--------|
| 23 | Training pipeline automation | TP-1 to TP-5 | 1 week |
| 24 | Shift reports + PDF export | RP-3, RP-4 | 1 week |
| 25 | Webhook / API integration | NT-3 | 2 days |
| 26 | Multi-camera scaling (25-50) | BE-2 | 1 week |
| 27 | Binary packaging (PyInstaller + PyArmor) | LIC-9 | 3 days |
| 28 | On-site deployment + tuning | — | 2 weeks buffer |

---

## 5. Dependencies & Risks

| Dependency | Impact | Mitigation |
|------------|--------|------------|
| TMEIC camera footage for training | Blocks all custom detection models (CF-1 to CF-8) | Start with public datasets + Roboflow models. Fine-tune when footage arrives. |
| TMEIC camera access (RTSP) | Blocks on-site testing | Test with recorded footage first. RTSP integration ready before site visit. |
| Jetson / DGX Spark hardware | Blocks production deployment | Develop on dev machine (MPS), TensorRT export tested separately. |
| PLC/speaker hardware specs | Blocks PLC integration | Get PLC model and protocol from TMEIC early. |
| Motor/drive monitoring requirements | CF-3 is undefined | Schedule call with TMEIC engineering to define "normal" vs "abnormal" state. |
| Domain + VPS for license server | Blocks product key activation | Register `license.safetylens.ai`, set up VPS. Low cost (< $15/month). |
| RSA key pair generation | Blocks license signing | Generate RSA-2048 key pair. Private key secured (never on customer device). |

---

## 6. Out of Scope (This Release)

- Face recognition for PPE violators
- Emergency boundary crossing (needs clarification)
- Multi-site management
- Cloud deployment option
- ERP/SAP deep integration (webhook only for now)
- Snake detection (TBD-2, feasibility unclear)
- Drugs/syringe detection (TBD-3, not feasible with standard cameras)

---

## 7. Success Criteria

| Metric | Target |
|--------|--------|
| Detection accuracy (PPE, phone, zone intrusion) | >= 90% on TMEIC footage |
| Detection accuracy (custom features) | >= 85% on TMEIC footage |
| Alert latency (detection to dashboard) | < 2 seconds |
| False positive rate | < 10% after 1 week of tuning |
| System uptime | 99% during working hours |
| Cameras supported (Base Pack) | 10 on Orin NX, 50 on AGX Orin |
| Inference FPS per camera | >= 5 FPS |
