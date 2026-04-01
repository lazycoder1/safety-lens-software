# PRD: Video Analytics Demo — Dual-Mode Detection

**Date:** 2026-03-05
**Status:** Draft
**Author:** Gautham

## Problem

We need a demo that clearly shows when to use traditional object detection (YOLO) vs vision-language models (Qwen VL), and how combining both creates a superior pipeline. The demo must work today with uploaded video files and switch to live RTSP camera feeds with zero code changes.

## Existing Foundation

We already have a working camera monitoring system at `/Users/gauthamgsabahit/workspace/scripts/tests-camera/` with:

- **Interface-based architecture** — `IStreamReader`, `ICameraStream`, `IObjectDetector`, `IAlertManager`
- **4 stream readers** already implemented:
  - `VideoFileReader` — for uploaded videos (demo mode)
  - `RTSPStreamReader` — for RTSP cameras (production mode)
  - `MJPEGStreamReader` — for Matrix SATATYA cameras
  - `WebcamStreamReader` — for local webcam
- **`CameraStream`** — unified interface with strategy pattern, selects reader via `stream_type` param
- **`YOLODetector`** — implements `IObjectDetector`, supports any YOLOv8/v11 model
- **`AlertManager`** — cooldown-based alerting on detected classes
- **CLI args** — `--stream-type file/rtsp/mjpeg/webcam`, `--model`, `--confidence`, etc.

### What's Missing

Only one analysis mode exists (YOLO). We need to add a second mode (VLM) and a combined mode.

## Requirements

### Core: Two Analysis Modes

#### Mode 1: Real-Time Detection (YOLO)
- Process every frame at 15-30+ FPS
- Output: annotated video with bounding boxes, class labels, confidence scores
- Counts per class, FPS overlay
- Alerts on target class detection
- **Already implemented** — just needs minor UI tweaks for side-by-side

#### Mode 2: Intelligent Analysis (Qwen VL)
- Sample frames from the stream (configurable: every N seconds or on trigger)
- Send frame batch to Qwen3-VL via Ollama API
- Output: natural language analysis — what's happening, safety issues, anomalies
- Structured JSON output option for downstream integration
- **New — needs implementation**

#### Mode 3: Combined (stretch goal for demo)
- YOLO runs real-time on every frame
- When YOLO detects an interesting event (new object, anomaly, alert trigger), crop the region
- Send crop + context to Qwen VL for deeper analysis
- Output: real-time boxes + periodic natural language insights

### Input Sources

| Source | Config | Demo Day | Production |
|--------|--------|----------|------------|
| Video file (MP4) | `--stream-type file --video-file path.mp4` | Primary | Testing |
| RTSP stream | `--stream-type rtsp --camera-ip x.x.x.x` | If camera available | Primary |
| Webcam | `--stream-type webcam` | Backup | Dev/testing |

**Key constraint:** Switching between video file and RTSP must require ONLY changing CLI args, no code changes. This is already satisfied by the existing `CameraStream` strategy pattern.

### Video Specs (for demo files)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Format | MP4 (H.264 codec) | Works with OpenCV + Qwen VL |
| Resolution | 720p (1280x720) | Good balance for local processing |
| FPS | 24-30 | Natural for YOLO, Qwen VL will sample down |
| Duration | 30-60 seconds | Long enough to show both modes |
| Audio | Strip it (`ffmpeg -an`) | Not needed, saves space |
| Content | Factory/warehouse with workers, equipment, PPE | Shows safety use case |

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Input Layer                       │
│  (VideoFileReader | RTSPStreamReader | Webcam)       │
│  ── all implement IStreamReader ──                   │
│  ── selected via CameraStream strategy ──            │
└──────────────────┬──────────────────────────────────┘
                   │ frames (np.ndarray BGR)
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ┌──────────────┐  ┌──────────────────┐
   │  Mode 1:     │  │  Mode 2:         │
   │  YOLODetector│  │  VLMAnalyzer     │
   │  (every frame│  │  (sampled frames)│
   │   ~30 FPS)   │  │  (every N sec)   │
   │              │  │                  │
   │  implements  │  │  implements      │
   │  IObjectDet. │  │  IAnalyzer (new) │
   └──────┬───────┘  └──────┬───────────┘
          │                  │
          ▼                  ▼
   Bounding boxes     Natural language
   + counts           analysis + alerts
          │                  │
          └────────┬─────────┘
                   ▼
            ┌──────────────┐
            │  Output Layer │
            │  - CV2 window │
            │  - Console    │
            │  - JSON/API   │
            └──────────────┘
```

### New Interface: IAnalyzer

```python
class IAnalyzer(ABC):
    """Interface for frame/video analysis beyond object detection"""

    @abstractmethod
    def analyze_frame(self, frame: np.ndarray, prompt: str = "") -> AnalysisResult:
        """Analyze a single frame and return natural language result"""
        pass

    @abstractmethod
    def analyze_frames(self, frames: List[np.ndarray], prompt: str = "") -> AnalysisResult:
        """Analyze a batch of frames (video segment)"""
        pass

@dataclass
class AnalysisResult:
    summary: str              # Natural language summary
    observations: List[str]   # Individual observations
    safety_issues: List[str]  # Safety-specific findings
    confidence: float         # Overall confidence
    latency_ms: float         # Processing time
    raw_response: str         # Full model response
```

### New Implementation: QwenVLAnalyzer

```python
class QwenVLAnalyzer(IAnalyzer):
    """Qwen3-VL based analyzer via Ollama API"""

    def __init__(self, model: str = "qwen3-vl:8b", ollama_url: str = "http://localhost:11434"):
        ...

    def analyze_frame(self, frame: np.ndarray, prompt: str = "") -> AnalysisResult:
        # 1. Encode frame to base64
        # 2. Call Ollama /api/chat with image
        # 3. Parse response into AnalysisResult
        ...

    def analyze_frames(self, frames: List[np.ndarray], prompt: str = "") -> AnalysisResult:
        # 1. Encode frames
        # 2. Send as multi-image or video context
        # 3. Parse response
        ...
```

### CLI Changes

New args to add to `main.py`:

```
--mode yolo|vlm|combined     Analysis mode (default: yolo)
--vlm-model qwen3-vl:8b     VLM model for Ollama (default: qwen3-vl:8b)
--vlm-interval 5            Seconds between VLM analyses (default: 5)
--vlm-prompt "..."          Custom analysis prompt
--ollama-url http://...     Ollama server URL (default: http://localhost:11434)
```

### Demo Flow

```bash
# Step 1: YOLO mode — show real-time detection
python main.py --stream-type file --video-file factory.mp4 --mode yolo

# Step 2: VLM mode — show intelligent analysis
python main.py --stream-type file --video-file factory.mp4 --mode vlm

# Step 3: Combined mode — show the power of both
python main.py --stream-type file --video-file factory.mp4 --mode combined

# Step 4: Switch to live camera (when available) — same code, different source
python main.py --stream-type rtsp --camera-ip 192.168.29.250 --mode combined
```

## Non-Goals (for now)

- Web UI / dashboard (console + CV2 window is enough for demo)
- Multi-camera support (single stream for demo)
- Cloud deployment
- Model training / fine-tuning
- Recording / storage of analysis results
- Authentication / multi-tenancy

## Technical Constraints

### Local Hardware: Mac M1 Pro 32GB

| Component | Resource Usage |
|-----------|---------------|
| YOLO (yolov8n) | ~200 MB RAM, MPS accelerated |
| Qwen3-VL-8B (Q4) | ~5 GB RAM via Ollama |
| OpenCV frame buffer | ~50 MB |
| **Total** | **~6 GB** — well within 32GB |

### Dependencies

```
# Existing
opencv-python
ultralytics
numpy
requests

# New
ollama              # Python client for Ollama API
```

### Ollama Setup (prerequisite)

```bash
brew install ollama
ollama serve                  # start server
ollama pull qwen3-vl:8b      # pull model (~5GB)
```

## Implementation Plan

### Phase 1: VLM Analyzer (Day 1)
1. Add `IAnalyzer` interface to `src/interfaces/`
2. Implement `QwenVLAnalyzer` in `src/impl/`
3. Add `--mode` CLI arg
4. Wire up VLM mode in main loop — sample every N seconds, print analysis to console

### Phase 2: Combined Mode (Day 1-2)
1. Run YOLO on every frame
2. On alert trigger or interval, send frame to VLM
3. Display both: annotated video (YOLO) + text overlay (VLM analysis)

### Phase 3: Demo Polish (Day 2)
1. Side-by-side display: YOLO annotated frame | VLM analysis text panel
2. Download/record a good factory safety video
3. Test with both video file and RTSP (if camera available)
4. Record a GIF/video of the demo

### Phase 4: RTSP Production (Future)
1. Already supported — just `--stream-type rtsp --camera-ip x.x.x.x`
2. Add reconnection logic for dropped RTSP streams
3. Add frame buffering for VLM to handle network jitter
4. Webhook/API output for alerts

## Success Criteria

- [ ] Upload a factory video, run in YOLO mode — see real-time annotated output
- [ ] Same video, run in VLM mode — see natural language safety analysis
- [ ] Same video, run in combined mode — see both working together
- [ ] Change `--stream-type file` to `--stream-type rtsp` — same analysis, live feed
- [ ] Clear visual difference between modes in the demo
- [ ] Total processing fits within 32GB RAM locally

## Open Questions

1. **Which Qwen VL model for demo?** Recommendation: `qwen3-vl:8b` — best balance of speed and quality on 32GB. Could also try `qwen3.5:9b` (built-in vision, newer).
2. **Structured output?** Should VLM return JSON (for downstream use) or free-text (more impressive in demo)? Recommendation: both — structured internally, rendered as text in demo.
3. **Frame sampling strategy for VLM?** Options: fixed interval (every 5s), on YOLO trigger (new detection), or motion-based. Recommendation: start with fixed interval, add trigger-based in Phase 2.
