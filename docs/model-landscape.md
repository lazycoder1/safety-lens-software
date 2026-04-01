# Vision Model Landscape (March 2026)

## Qwen VL Model Family

### Qwen2.5-VL (Jan 2025)
- **Sizes:** 3B, 7B, 32B, 72B
- **Status:** Mature, widely supported
- **Highlights:** Matches GPT-4o/Claude 3.5 Sonnet on document understanding (72B). 7B outperforms GPT-4o-mini.
- **Ollama:** `ollama run qwen2.5vl:7b`

### Qwen3-VL (Oct 2025)
- **Sizes:** 2B, 4B, 8B, 32B (dense) + MoE variants
- **Status:** Best dedicated VL model in the Qwen family
- **Highlights:**
  - Native 256K context, expandable to 1M
  - Hours-long video comprehension with full recall
  - GUI/agent interaction — can operate computer and mobile interfaces
  - Code generation from images/videos (mockup to HTML/CSS/JS)
  - Top performance on OS World benchmark
- **Ollama:** `ollama run qwen3-vl:8b` (requires Ollama v0.12.7+)

### Qwen3.5 (Feb 2026)
- **Sizes:** 4B, 9B, 27B, 35B-A3B (MoE), 397B-A17B (MoE)
- **Status:** Latest generation
- **Key change:** Vision is built-in via early fusion — there is NO separate "Qwen3.5-VL" model
- **Highlights:**
  - Trained on trillions of multimodal tokens
  - Outperforms Qwen3-VL across reasoning, coding, agents, and visual understanding
  - 201 language support
  - MoE variants are efficient (35B-A3B = only 3B active params)
- **Ollama:** `ollama run qwen3.5:9b`

## YOLO Family

### YOLOv8 / YOLOv11 (Ultralytics)
- **Sizes:** nano (n), small (s), medium (m), large (l), xlarge (x)
- **Model weights:** < 1 GB even for largest variants
- **Speed:** 100+ FPS real-time detection
- **Install:** `pip install ultralytics`
- **Open-vocab variant:** YOLO-World (prompt-then-detect paradigm)

### Other Detection Models Worth Knowing
- **RT-DETR:** Real-time transformer-based detection (Meta)
- **RF-DETR:** Roboflow's DETR variant, strong on small objects
- **GroundingDINO:** Zero-shot detection with text prompts — outperforms Qwen2.5-VL-72B on zero-shot detection despite being much smaller
- **SAM3:** Segment Anything Model v3 for segmentation tasks

## Key Insight: Specialist vs Generalist

| Dimension | YOLO | Qwen VL |
|-----------|------|---------|
| **Task** | Detection, counting, tracking | Visual reasoning, understanding, QA |
| **Speed** | Real-time (100+ FPS) | 1-5 seconds per image |
| **Training needed?** | Yes, fine-tune on your classes | No — zero-shot via prompting |
| **Output** | Bounding boxes + class labels + confidence | Natural language, structured data |
| **Edge deployment** | Excellent | Difficult (need quantized small models) |
| **Complex reasoning** | No | Yes — "why is this unsafe?" |
| **Video** | Frame-by-frame, real-time | Can understand full video narrative |

**Rule of thumb:**
- YOLO = "WHAT is where?" (fast, trained, precise)
- Qwen VL = "WHAT does this mean?" (slow, zero-shot, reasoning)

## References
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3.5 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Qwen3-VL on Ollama](https://ollama.com/library/qwen3-vl)
- [Object Detection with Qwen2.5-VL — LearnOpenCV](https://learnopencv.com/object-detection-with-vlms-ft-qwen2-5-vl/)
- [Qwen VL Max vs YOLO World — Roboflow](https://playground.roboflow.com/models/compare/qwen-vl-max-vs-yolo-world)
