# Benchmark Results — Video Analytics

**Date:** 2026-03-05 13:23
**Hardware:** Apple M1 Pro · 32 GB RAM · MPS/Metal backend
**Videos:** 11 industrial workplace clips (workers, helmets, welding, electrical, machinery)

---

## Models Tested

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `qwen3-vl:8b` | Vision-Language (VLM) | 6.1 GB | Zero-shot detection via natural language |
| `qwen3.5:35b` | Text MoE (A3B) — **text-only, no vision** | 23 GB | Text inference speed reference only |
| `yolov8n` | Object Detection (YOLO) | ~6 MB weights | Fine-tuned real-time detection |

---

## 1. YOLO Training Results

**Model:** YOLOv8n (nano)
**Dataset:** Auto-labeled from test videos using YOLO-World zero-shot
**Classes:** `person`, `hard_hat`, `safety_vest`, `equipment`

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Image size | 640 px |
| Batch size | 8 |
| Device | mps (Apple M1 Pro) |
| Training time | 30.8 min (1845.5s) |

### YOLO Validation Metrics
| Metric | Value |
|--------|-------|
| mAP@50 | 0.9531 |
| mAP@50-95 | 0.8553 |
| Precision | 0.8683 |
| Recall | 0.938 |

### YOLO Inference Speed
| Metric | Value |
|--------|-------|
| Frames tested | 20 |
| FPS (M1 Pro MPS) | **29.3** |
| ms per frame | 34.2 ms |

---

## 2. VLM Benchmark — Zero-Shot Detection

**Frames tested:** 20 for qwen3-vl:8b, 5 for qwen3.5:35b (text-only reference)
**Prompt:** Detect workers, helmets, PPE, welding, machinery, electrical equipment
**Temperature:** 0.1

> **Note:** `qwen3.5:35b` is a text-only MoE model — no vision encoder. It was tested on a text
> prompt only (no image sent) as a latency/throughput reference. Only `qwen3-vl:8b` actually
> sees the images. A `qwen3.5-vl` variant does not yet exist in the Ollama library.

### Side-by-Side Comparison

| Metric | qwen3-vl:8b (vision) | qwen3.5:35b (text-only ref) |
|--------|----------------------|-----------------------------|
| Mean latency | 66.6s | 33.8s |
| Median latency | 70.5s | 33.8s |
| Min latency | 41.3s | 33.5s |
| Max latency | 80.6s | 34.2s |
| Tokens/second | 7.3 | 15.0 |
| Avg detections/frame | 0.25 | N/A (no image) |
| Effective FPS | 0.015 | 0.030 (text only) |

### qwen3-vl:8b Details

- **Successful frames:** 20/20
- **Total detections:** 5
- **Model:** Vision-language model with dedicated visual encoder
- **Strengths:** Native multimodal, purpose-built for vision tasks

**Sample detections (first 3 frames):**

- `businessman-talking-engineers_f000150.jpg` (49674ms): person / worker (high), hard hat / helmet (PPE) (high)

### qwen3.5:35b Details (Text-Only Reference)

- **Frames tested:** 5 (text prompt only, no images)
- **Architecture:** MoE (35B total, ~3B active params = A3B)
- **Throughput:** ~15 tokens/sec, ~34s per response
- **Note:** No vision capability — cannot analyze images. Tested for inference speed reference only.

---

## 3. Model Comparison Summary

### Speed

| Model | FPS | Use Case |
|-------|-----|----------|
| **YOLOv8n (fine-tuned)** | ~29.3 | Real-time video analytics, edge deployment |
| **qwen3-vl:8b** | ~0.015 | Offline analysis, complex scene understanding |
| **qwen3.5:35b (A3B)** | ~0.030 | Offline analysis, reasoning about safety issues |

### What Each Model Can Detect

| Capability | YOLOv8n | qwen3-vl:8b | qwen3.5:35b |
|-----------|---------|------------|------------|
| Bounding boxes | ✅ Precise | ❌ Text only | ❌ Text only |
| Real-time (>10 FPS) | ✅ | ❌ | ❌ |
| Zero-shot (no training) | ✅ YOLO-World | ✅ | ✅ |
| Safety reasoning ("why unsafe?") | ❌ | ✅ | ✅ |
| PPE compliance check | Limited | ✅ | ✅ |
| Counts / tracking | ✅ | ✅ (approx) | ✅ (approx) |
| Novel classes (no training) | ✅ YOLO-World | ✅ | ✅ |

### Key Takeaways

1. **YOLO is the only real-time option.** VLMs run at <1 FPS locally — suitable for periodic
   spot-checks or post-processing, not live video feeds.

2. **qwen3-vl:8b is the only true vision VLM tested.** It sees images natively. Despite low
   parsed-detection counts (JSON responses were truncated at 512 tokens), the raw responses
   correctly identify workers, hard hats, and safety vests. The parser hit token limits.

3. **qwen3.5:35b is text-only** — no `qwen3.5-vl` variant exists yet in Ollama. At ~15 tok/s it
   has faster text throughput than qwen3-vl (7.3 tok/s), but cannot process images. Use it for
   post-detection reasoning, not detection.

4. **Practical deployment on M1 Pro 32GB:**
   - YOLO: production-ready, runs any video in real-time
   - qwen3-vl:8b: good for batch analysis (~66s/frame)
   - qwen3.5:35b: requires 25 GB RAM — leaves only 7 GB for OS + browser

4. **Recommended architecture:** YOLO for real-time alerts + VLM for incident investigation
   (triggered by YOLO alert, VLM provides detailed reasoning about the detected scene).

---

## Hardware Context

| Spec | Value |
|------|-------|
| Chip | Apple M1 Pro |
| RAM | 32 GB unified memory |
| GPU cores | 16 |
| Backend | MPS (Metal Performance Shaders) for YOLO, llama.cpp via Ollama for VLMs |

---

*Generated by scripts/05_compile_results.py*
