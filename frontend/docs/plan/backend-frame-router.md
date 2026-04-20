# Backend Frame Router — Stream & Resolution Management

## Overview

The frontend configures "what rules on what cameras." The backend computes the optimal stream strategy — which RTSP stream to pull, at what FPS, and which frames go to which models. Admins should not need to manually configure resolution or FPS per rule.

---

## RTSP Stream Profiles

Most IP cameras (Hikvision, Dahua, Axis) serve two simultaneous RTSP streams:

| Profile | Typical Resolution | Typical FPS | Use |
|---------|-------------------|-------------|-----|
| Main stream | 1080p / 4K | 25 fps | Recording, VLM analysis, high-res detection |
| Sub stream | 480p / 720p | 15 fps | Real-time YOLO detection |

Both streams are served simultaneously at no extra camera-side cost. The backend decides which to pull per camera based on assigned rules.

### Camera Config (stored in DB, configured via UI)

```json
{
  "id": "c5",
  "name": "Battery Room - Charging",
  "main_rtsp": "rtsp://admin:pass@192.168.1.105:554/stream1",
  "sub_rtsp": "rtsp://admin:pass@192.168.1.105:554/stream2",
  "zone": "Battery Room",
  "fps_override": null
}
```

- `main_rtsp`: High-res stream URL. Pulled only when needed.
- `sub_rtsp`: Low-res stream URL. Always pulled for real-time detection.
- `fps_override`: Optional. Admin can cap FPS if Jetson is overloaded. `null` = auto.

---

## Rule Sampling Defaults

Each rule has a sampling strategy auto-assigned by the backend based on model type and category. These are defaults — can be overridden per-rule in advanced config.

### Sampling Modes

| Mode | Description | Config |
|------|-------------|--------|
| `every_frame` | Process every decoded frame | `{"mode": "every_frame"}` |
| `every_n_frames` | Process every Nth frame | `{"mode": "every_n", "n": 3}` |
| `interval` | Process one frame every N seconds | `{"mode": "interval", "seconds": 60}` |
| `triggered` | Only run when another rule fires | `{"mode": "triggered", "trigger_rule": "r3"}` |

### Default Auto-Assignment

| Rule Category | Model | Stream | Resolution | Sampling | Rationale |
|--------------|-------|--------|-----------|----------|-----------|
| Zone intrusion | YOLO26 | sub | 720p | every_frame | Need real-time person tracking for geofence |
| Person fall | YOLO-pose | sub | 720p | every_frame | Need motion continuity for pose change detection |
| Mobile phone | YOLO26 | sub | 720p | every_frame | Phone in hand is transient, need real-time |
| Helmet/vest (PPE) | YOLOE text | sub | 720p | every_3rd_frame | PPE state doesn't change frame-to-frame |
| Safety goggles | YOLOE text | sub | 720p | every_3rd_frame | Same as above |
| Safety harness | YOLOE visual | sub | 720p | every_3rd_frame | Same as above |
| Head cap vs helmet | YOLOE text | sub | 720p | every_3rd_frame | Same as above |
| Forklift operator | YOLOE text | sub | 720p | every_3rd_frame | Same as above |
| Animal detection | YOLOE text | sub | 720p | interval 5s | Animals move slowly, 5s is sufficient |
| Fire/smoke | YOLOE text | sub | 720p | every_frame | Safety-critical, need fast detection |
| Drugs/syringes | YOLOE text | main | 1080p | interval 10s | Small objects need resolution |
| Gangway blockage | VLM | main | 1080p | interval 60s | Scene reasoning, not real-time |
| VLM escalation | VLM | main | 1080p | triggered | Only fires on YOLOE low-confidence |
| Incident investigation | VLM | main | 1080p | triggered | Only fires after any P1/P2 alert |

---

## Frame Router — Core Algorithm

The Frame Router is a Python service that runs on startup and recomputes whenever camera/rule config changes.

### Step 1: Compute Stream Plan Per Camera

```python
from dataclasses import dataclass

@dataclass
class StreamPlan:
    camera_id: str
    sub_stream_url: str
    sub_stream_decode_fps: float      # How fast we decode the sub stream
    main_stream_url: str | None       # None if no rule needs main stream
    main_stream_decode_fps: float     # 0 if not needed
    frame_routes: list[FrameRoute]

@dataclass
class FrameRoute:
    rule_id: str
    model: str                        # "yolo26", "yoloe", "yolo-pose", "vlm"
    stream: str                       # "sub" or "main"
    sampling: dict                    # {"mode": "every_frame"} etc.
    min_resolution: int               # 480, 720, 1080


def compute_stream_plan(camera_id: str, rules: list[Rule]) -> StreamPlan:
    """Compute optimal stream strategy for a camera based on assigned rules."""

    frame_routes = []
    needs_main_stream = False
    max_sub_fps = 0
    max_main_fps = 0

    for rule in rules:
        route = FrameRoute(
            rule_id=rule.id,
            model=rule.model,
            stream="main" if rule.min_resolution > 720 else "sub",
            sampling=rule.sampling,
            min_resolution=rule.min_resolution,
        )
        frame_routes.append(route)

        if route.stream == "main":
            needs_main_stream = True
            if route.sampling["mode"] == "every_frame":
                max_main_fps = max(max_main_fps, 15)
            elif route.sampling["mode"] == "interval":
                max_main_fps = max(max_main_fps, 1.0 / route.sampling["seconds"])
        else:
            if route.sampling["mode"] == "every_frame":
                max_sub_fps = max(max_sub_fps, 15)
            elif route.sampling["mode"] == "every_n":
                max_sub_fps = max(max_sub_fps, 15)  # Still decode at full, skip in routing

    return StreamPlan(
        camera_id=camera_id,
        sub_stream_url=camera.sub_rtsp,
        sub_stream_decode_fps=max_sub_fps or 10,  # Default 10fps
        main_stream_url=camera.main_rtsp if needs_main_stream else None,
        main_stream_decode_fps=max_main_fps,
        frame_routes=frame_routes,
    )
```

### Step 2: Stream Decoder (per camera)

```python
import cv2
import time
from threading import Thread
from queue import Queue

class StreamDecoder(Thread):
    """Decodes RTSP stream at target FPS, pushes frames to routing queue."""

    def __init__(self, rtsp_url: str, target_fps: float, frame_queue: Queue):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.frame_queue = frame_queue
        self.interval = 1.0 / target_fps if target_fps > 0 else float('inf')
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        last_decode_time = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Reconnect logic
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            now = time.monotonic()
            if now - last_decode_time >= self.interval:
                last_decode_time = now
                self.frame_queue.put((now, frame))

        cap.release()
```

### Step 3: Frame Router (dispatches frames to models)

```python
class FrameRouter:
    """Routes decoded frames to appropriate model inference queues."""

    def __init__(self, stream_plan: StreamPlan):
        self.plan = stream_plan
        self.frame_counters = {}  # rule_id -> frame count since last process
        self.last_interval = {}   # rule_id -> last process timestamp

    def route_frame(self, timestamp: float, frame, stream: str):
        """Decide which rules should process this frame."""

        tasks = []

        for route in self.plan.frame_routes:
            if route.stream != stream:
                continue

            should_process = False
            sampling = route.sampling

            if sampling["mode"] == "every_frame":
                should_process = True

            elif sampling["mode"] == "every_n":
                counter = self.frame_counters.get(route.rule_id, 0) + 1
                self.frame_counters[route.rule_id] = counter
                if counter >= sampling["n"]:
                    self.frame_counters[route.rule_id] = 0
                    should_process = True

            elif sampling["mode"] == "interval":
                last = self.last_interval.get(route.rule_id, 0)
                if timestamp - last >= sampling["seconds"]:
                    self.last_interval[route.rule_id] = timestamp
                    should_process = True

            elif sampling["mode"] == "triggered":
                pass  # Handled by alert system, not frame router

            if should_process:
                # Resize frame if needed (e.g., sub stream is 720p but rule only needs 480p)
                processed_frame = self._maybe_resize(frame, route.min_resolution)
                tasks.append((route.rule_id, route.model, processed_frame))

        return tasks

    def _maybe_resize(self, frame, target_resolution: int):
        """Downscale frame if it's larger than what the rule needs."""
        h = frame.shape[0]
        if h > target_resolution:
            scale = target_resolution / h
            return cv2.resize(frame, None, fx=scale, fy=scale)
        return frame
```

### Step 4: Model Inference Pool

```python
class InferencePool:
    """Manages model instances and batches inference requests."""

    def __init__(self):
        self.yolo26 = YOLO("yolo26n.engine")       # TensorRT on Jetson
        self.yoloe = YOLOE("yoloe.engine")          # TensorRT on Jetson
        self.yolo_pose = YOLO("yolo26n-pose.engine") # TensorRT on Jetson
        # VLM is a cloud API call, not a local model

    async def infer(self, model: str, frame, rule_id: str):
        """Run inference and return detections."""

        if model == "yolo26":
            results = self.yolo26.predict(frame, conf=0.5)
            return self._parse_yolo_results(results, rule_id)

        elif model == "yoloe":
            # YOLOE uses text prompts loaded from rule config
            rule_config = get_rule_config(rule_id)
            self.yoloe.set_classes(rule_config.prompts)
            results = self.yoloe.predict(frame, conf=rule_config.confidence)

            # Check for low confidence → escalate to VLM
            detections = self._parse_yolo_results(results, rule_id)
            for det in detections:
                if det.confidence < rule_config.vlm_escalation_threshold:
                    await self._escalate_to_vlm(frame, det, rule_id)

            return detections

        elif model == "yolo-pose":
            results = self.yolo_pose.predict(frame, conf=0.5)
            return self._check_fall(results, rule_id)

        elif model == "vlm":
            # Cloud API call
            rule_config = get_rule_config(rule_id)
            return await self._call_vlm(frame, rule_config.prompt, rule_id)

    async def _escalate_to_vlm(self, frame, detection, rule_id):
        """YOLOE low-confidence detection → send to VLM for confirmation."""
        rule_config = get_rule_config(rule_id)
        prompt = f"I detected a possible '{detection.class_name}' with {detection.confidence:.0%} confidence. Is this correct? Is there a safety violation?"
        vlm_result = await self._call_vlm(frame, prompt, rule_id)
        if vlm_result.confirms_violation:
            emit_alert(rule_id, detection, vlm_reasoning=vlm_result.text)
```

---

## GPU Budget Estimation

### Per-Camera Inference Cost (approximate, Jetson Orin Nano)

| Model | Resolution | Per-Frame Time | At 10 FPS | GPU % (of 40 TOPS) |
|-------|-----------|---------------|-----------|-------------------|
| YOLO26n | 720p | ~8ms | 80ms/s | ~8% |
| YOLOE (text prompt) | 720p | ~12ms | 120ms/s* | ~12% |
| YOLO26n-pose | 720p | ~10ms | 100ms/s | ~10% |
| Total per camera (all models) | | | | ~20-25% |

*YOLOE runs every 3rd frame, so actual cost is ~40ms/s = ~4%

### Budget Calculator

```python
def estimate_gpu_budget(cameras: list[Camera], rules: list[Rule]) -> float:
    """Estimate total GPU utilization percentage."""

    MODEL_COST_MS = {  # Per frame at 720p
        "yolo26": 8,
        "yoloe": 12,
        "yolo-pose": 10,
    }

    total_ms_per_second = 0

    for camera in cameras:
        camera_rules = [r for r in rules if camera.id in r.camera_ids]
        plan = compute_stream_plan(camera.id, camera_rules)

        for route in plan.frame_routes:
            if route.model == "vlm":
                continue  # Cloud, no local GPU cost

            cost_per_frame = MODEL_COST_MS.get(route.model, 10)

            if route.sampling["mode"] == "every_frame":
                fps = plan.sub_stream_decode_fps
            elif route.sampling["mode"] == "every_n":
                fps = plan.sub_stream_decode_fps / route.sampling["n"]
            elif route.sampling["mode"] == "interval":
                fps = 1.0 / route.sampling["seconds"]
            else:
                fps = 0

            total_ms_per_second += cost_per_frame * fps

    # Jetson Orin Nano: ~1000ms available per second on GPU
    gpu_utilization = (total_ms_per_second / 1000) * 100
    return gpu_utilization
```

### Capacity Guidelines

| Jetson Device | Budget | Cameras @ 10fps (YOLO26 + YOLOE) |
|--------------|--------|----------------------------------|
| Orin Nano 8GB | 40 TOPS | 2-4 cameras |
| Orin NX 16GB | 100 TOPS | 5-8 cameras |
| AGX Orin | 275 TOPS | 15-20 cameras |

---

## Auto-Scaling: What Happens When GPU is Overloaded

If GPU utilization exceeds a threshold (default 85%), the backend auto-degrades:

### Degradation Priority (shed lowest priority first)

1. **Reduce YOLOE sampling** — every 3rd frame → every 5th frame
2. **Reduce PPE rule FPS** — helmets/vests detected at 2fps instead of 5fps (PPE state is slow-changing)
3. **Reduce sub stream decode FPS** — 10fps → 7fps → 5fps
4. **Skip animal detection** — lowest priority, can pause
5. **Never degrade**: fall detection, fire/smoke, zone intrusion (safety-critical)

```python
DEGRADATION_PRIORITY = [
    # (rule_category, action)
    ("environment", "reduce_sampling"),     # Animal, gangway → less frequent
    ("PPE", "reduce_sampling"),             # Helmet, vest → every 5th frame
    ("behavior", "reduce_sampling"),        # Phone usage → every 5th frame
    ("ALL", "reduce_decode_fps"),           # Global FPS reduction
    # NEVER degrade:
    # - "emergency" (fall, fire)
    # - "zone_safety" (intrusion)
]
```

### Auto-Recovery

When GPU drops below 70%, gradually restore sampling rates back to defaults.

---

## Config Change Flow

```
Admin changes rule config in UI
        |
        v
Frontend sends PUT /api/rules/{id}
        |
        v
Backend receives config change
        |
        v
Frame Router recomputes stream plans for affected cameras
        |
        v
Stream decoders updated (add/remove main stream, adjust FPS)
        |
        v
No restart needed — hot-reload of frame routing
```

---

## What the UI Needs to Show

The UI does NOT expose resolution/FPS per rule. Instead:

1. **Camera config**: main + sub RTSP URLs (auto-detected via ONVIF if possible)
2. **System > Models page**: GPU utilization gauge, per-camera inference cost breakdown
3. **Warning banner**: if estimated GPU > 85%, show "Consider reducing cameras or upgrading to Orin NX"
4. **Advanced override** (admin only): per-camera FPS cap, per-rule sampling override — hidden behind "Advanced" toggle, rarely needed
