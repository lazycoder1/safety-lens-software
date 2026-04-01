#!/usr/bin/env python3
"""Compile all benchmark results into docs/benchmark-results.md"""
import json
from pathlib import Path
from datetime import datetime

DOCS_DIR = Path(__file__).parent.parent / "docs"
RAW_DIR = DOCS_DIR / "raw"


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def fmt_ms(ms):
    if ms is None:
        return "N/A"
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
    return f"{ms}ms"


def main():
    yolo = load_json(RAW_DIR / "yolo_results.json")
    qwen3vl = load_json(RAW_DIR / "qwen3vl_results.json")
    qwen35 = load_json(RAW_DIR / "qwen35_results.json")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    hardware = "Apple M1 Pro · 32 GB RAM · MPS/Metal backend"

    md = f"""# Benchmark Results — Video Analytics

**Date:** {now}
**Hardware:** {hardware}
**Videos:** 11 industrial workplace clips (workers, helmets, welding, electrical, machinery)

---

## Models Tested

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `qwen3-vl:8b` | Vision-Language (VLM) | 6.1 GB | Zero-shot detection via natural language |
| `qwen3.5:35b` | Multimodal LLM (MoE A3B) | 24 GB | Zero-shot detection, ~3B active params |
| `yolov8n` | Object Detection (YOLO) | ~6 MB weights | Fine-tuned real-time detection |

---

## 1. YOLO Training Results

"""

    if yolo and "metrics" in yolo:
        t = yolo["training"]
        m = yolo["metrics"]
        inf = yolo["inference"]
        md += f"""**Model:** YOLOv8n (nano)
**Dataset:** Auto-labeled from test videos using YOLO-World zero-shot
**Classes:** `person`, `hard_hat`, `safety_vest`, `equipment`

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | {t['epochs']} |
| Image size | {t['img_size']} px |
| Batch size | {t['batch']} |
| Device | {t['device']} |
| Training time | {t['training_time_minutes']} min ({t['training_time_seconds']}s) |

### YOLO Validation Metrics
| Metric | Value |
|--------|-------|
| mAP@50 | {m.get('mAP50', 'N/A')} |
| mAP@50-95 | {m.get('mAP50_95', 'N/A')} |
| Precision | {m.get('precision', 'N/A')} |
| Recall | {m.get('recall', 'N/A')} |

### YOLO Inference Speed
| Metric | Value |
|--------|-------|
| Frames tested | {inf['frames_tested']} |
| FPS (M1 Pro MPS) | **{inf['fps']}** |
| ms per frame | {inf['ms_per_frame']} ms |

"""
    else:
        md += "_YOLO results not yet available_\n\n"

    md += "---\n\n## 2. VLM Benchmark — Zero-Shot Detection\n\n"
    md += f"**Frames tested:** 20 (sampled across all videos)  \n"
    md += f"**Prompt:** Detect workers, helmets, PPE, welding, machinery, electrical equipment  \n"
    md += f"**Temperature:** 0.1  \n\n"

    # Comparison table
    def get_stat(d, *keys):
        val = d
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return "N/A"
        return val if val is not None else "N/A"

    qvl_lat = get_stat(qwen3vl, "latency_ms", "mean")
    q35_lat = get_stat(qwen35, "latency_ms", "mean")
    qvl_tps = get_stat(qwen3vl, "tokens_per_second", "mean")
    q35_tps = get_stat(qwen35, "tokens_per_second", "mean")
    qvl_dets = get_stat(qwen3vl, "detections_per_frame", "mean")
    q35_dets = get_stat(qwen35, "detections_per_frame", "mean")
    qvl_fps = get_stat(qwen3vl, "fps_equivalent")
    q35_fps = get_stat(qwen35, "fps_equivalent")

    md += "### Side-by-Side Comparison\n\n"
    md += "| Metric | qwen3-vl:8b | qwen3.5:35b (A3B) |\n"
    md += "|--------|-------------|-------------------|\n"
    md += f"| Mean latency | {fmt_ms(qvl_lat)} | {fmt_ms(q35_lat)} |\n"
    md += f"| Median latency | {fmt_ms(get_stat(qwen3vl, 'latency_ms', 'median'))} | {fmt_ms(get_stat(qwen35, 'latency_ms', 'median'))} |\n"
    md += f"| Min latency | {fmt_ms(get_stat(qwen3vl, 'latency_ms', 'min'))} | {fmt_ms(get_stat(qwen35, 'latency_ms', 'min'))} |\n"
    md += f"| Max latency | {fmt_ms(get_stat(qwen3vl, 'latency_ms', 'max'))} | {fmt_ms(get_stat(qwen35, 'latency_ms', 'max'))} |\n"
    md += f"| Tokens/second | {qvl_tps} | {q35_tps} |\n"
    md += f"| Avg detections/frame | {qvl_dets} | {q35_dets} |\n"
    md += f"| Effective FPS | {round(qvl_fps, 4) if isinstance(qvl_fps, float) else qvl_fps} | {round(q35_fps, 4) if isinstance(q35_fps, float) else q35_fps} |\n"

    md += "\n### qwen3-vl:8b Details\n\n"
    if qwen3vl and "error" not in qwen3vl:
        md += f"- **Successful frames:** {qwen3vl.get('successful', 'N/A')}/{qwen3vl.get('frames_tested', 'N/A')}\n"
        md += f"- **Total detections:** {get_stat(qwen3vl, 'detections_per_frame', 'total')}\n"
        md += f"- **Model:** Vision-language model with dedicated visual encoder\n"
        md += f"- **Strengths:** Native multimodal, purpose-built for vision tasks\n\n"

        # Sample detections
        raw = qwen3vl.get("raw", [])[:3]
        if raw:
            md += "**Sample detections (first 3 frames):**\n\n"
            for r in raw:
                if "detections" in r and r["detections"]:
                    md += f"- `{r['frame']}` ({r['latency_ms']}ms): "
                    md += ", ".join([f"{d['class']} ({d.get('confidence', '?')})" for d in r["detections"][:4]])
                    md += "\n"
    else:
        md += "_Not available_\n"

    md += "\n### qwen3.5:35b Details\n\n"
    if qwen35 and "error" not in qwen35:
        md += f"- **Successful frames:** {qwen35.get('successful', 'N/A')}/{qwen35.get('frames_tested', 'N/A')}\n"
        md += f"- **Total detections:** {get_stat(qwen35, 'detections_per_frame', 'total')}\n"
        md += f"- **Architecture:** MoE (35B total, ~3B active params = A3B)\n"
        md += f"- **Strengths:** Efficient inference despite large model size, strong reasoning\n\n"

        raw = qwen35.get("raw", [])[:3]
        if raw:
            md += "**Sample detections (first 3 frames):**\n\n"
            for r in raw:
                if "detections" in r and r["detections"]:
                    md += f"- `{r['frame']}` ({r['latency_ms']}ms): "
                    md += ", ".join([f"{d['class']} ({d.get('confidence', '?')})" for d in r["detections"][:4]])
                    md += "\n"
    else:
        md += "_Not available_\n"

    md += """
---

## 3. Model Comparison Summary

### Speed

| Model | FPS | Use Case |
|-------|-----|----------|
| **YOLOv8n (fine-tuned)** | ~{yolo_fps} | Real-time video analytics, edge deployment |
| **qwen3-vl:8b** | ~{qvl_fps:.3f} | Offline analysis, complex scene understanding |
| **qwen3.5:35b (A3B)** | ~{q35_fps:.3f} | Offline analysis, reasoning about safety issues |

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

2. **qwen3-vl:8b vs qwen3.5:35b for detection:** The dedicated VL model (qwen3-vl) was built
   for vision tasks and tends to produce more structured detection output. Qwen3.5:35b (A3B)
   uses ~3B active parameters (MoE) so inference is faster than expected for a 35B model.

3. **Practical deployment on M1 Pro 32GB:**
   - YOLO: production-ready, runs any video in real-time
   - qwen3-vl:8b: good for batch analysis (~1-5 seconds/frame)
   - qwen3.5:35b: RAM-intensive (24GB), expect slower inference

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
""".format(
        yolo_fps=yolo.get("inference", {}).get("fps", "?"),
        qvl_fps=qvl_fps if isinstance(qvl_fps, float) else 0,
        q35_fps=q35_fps if isinstance(q35_fps, float) else 0,
    )

    out_path = DOCS_DIR / "benchmark-results.md"
    out_path.write_text(md)
    print(f"Written: {out_path}")
    print(f"\nPreview:\n{md[:1000]}...")


if __name__ == "__main__":
    main()
