# Demo Plan: YOLO vs Qwen VL — When to Use What

## Objective

Show the difference between specialist detection (YOLO) and generalist vision-language (Qwen VL) models, and demonstrate the power of combining both.

## Demo Structure

### 1. Speed vs Understanding

**Same image, two approaches:**

| Step | Model | What it does |
|------|-------|-------------|
| A | YOLOv8 | Instant bounding boxes — "3 persons, 2 helmets, 1 forklift" |
| B | Qwen3-VL-8B | "There are 3 workers near a forklift. One worker is not wearing a helmet, which is a safety violation." |

**Takeaway:** YOLO tells you WHAT. Qwen VL tells you WHY it matters.

### 2. Real-Time vs Batch

**Video feed demo:**

| Step | Model | What it does |
|------|-------|-------------|
| A | YOLOv8 | Process video at 30+ FPS, draw boxes in real-time |
| B | Qwen3-VL-8B | Analyze a 1-minute clip and summarize: "Worker entered restricted zone at 0:23, no PPE detected" |

**Takeaway:** YOLO for live monitoring. Qwen VL for post-event analysis.

### 3. Zero-Shot vs Trained

**Novel object detection:**

| Step | Model | What it does |
|------|-------|-------------|
| A | YOLOv8 (COCO-trained) | Cannot detect domain-specific objects it wasn't trained on |
| B | Qwen3-VL-8B | "I see a pressure gauge reading 45 PSI, which appears to be in the red zone" |
| C | YOLOv8 (fine-tuned) | After 30 min training on 200 labeled images: detects pressure gauges at 60 FPS |

**Takeaway:** Qwen VL for prototyping/zero-shot. YOLO when you need speed on known objects.

### 4. Combined Pipeline (Best of Both)

```
Camera Feed
    |
    v
[YOLOv8 @ 30 FPS] --> detections + crops
    |
    v
[Filter: interesting frames] --> confidence < threshold, new object, anomaly
    |
    v
[Qwen3-VL-8B] --> reasoning about the crop
    |
    v
[Alert / Report]
```

**Example flow:**
1. YOLO detects a person in a restricted zone (real-time)
2. Crop is sent to Qwen VL
3. Qwen VL: "Worker is not wearing required high-visibility vest. They appear to be carrying a toolbox, possibly performing unauthorized maintenance."
4. Alert generated with context

## Technical Setup (All Local)

```bash
# Terminal 1: Ollama server
ollama serve

# Terminal 2: Run demo
source .venv/bin/activate
python demo.py --source video.mp4 --yolo yolov8n.pt --vlm qwen3-vl:8b
```

## Comparison Summary Table (for slides)

| Criteria | YOLO | Qwen VL | Combined |
|----------|------|---------|----------|
| Speed | 100+ FPS | 1-5 sec/image | Real-time + selective deep analysis |
| Training needed | Yes | No | Only YOLO component |
| Output | Boxes + labels | Natural language | Structured alerts with context |
| Edge deploy | Yes | Difficult | YOLO on edge, VLM in cloud |
| Cost | Low | Higher (GPU/RAM) | Optimized — VLM only when needed |
| Novel objects | Needs retraining | Zero-shot | Zero-shot discovery, then train YOLO |

## Use Case Decision Matrix

| Use Case | Recommended Model |
|----------|------------------|
| Real-time counting/tracking | YOLO |
| Quality inspection (known defects) | YOLO (fine-tuned) |
| Document/receipt parsing | Qwen VL |
| Safety compliance analysis | Combined |
| Video summarization | Qwen VL |
| Anomaly detection (unknown anomalies) | Combined |
| Drone/robot navigation | YOLO |
| Customer behavior analysis | Combined |
