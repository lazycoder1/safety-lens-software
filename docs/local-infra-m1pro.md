# Local Infrastructure — Mac M1 Pro 32GB

## What Runs Locally

### Qwen VL Models (Inference Only)

| Model | Quantized Size | Feasibility | Command |
|-------|---------------|-------------|---------|
| Qwen3-VL-2B (Q4) | ~1.5 GB | Excellent | `ollama run qwen3-vl:2b` |
| Qwen3-VL-4B (Q4) | ~2.5 GB | Excellent | `ollama run qwen3-vl:4b` |
| Qwen3-VL-8B (Q4) | ~5 GB | Great | `ollama run qwen3-vl:8b` |
| Qwen3-VL-32B (Q4) | ~18 GB | Runs, tight on RAM | `ollama run qwen3-vl:32b` |
| Qwen3.5-9B (Q4) | ~6 GB | Great | `ollama run qwen3.5:9b` |
| Qwen3.5-27B (Q4) | ~16 GB | Good | `ollama run qwen3.5:27b` |
| Qwen3.5-35B-A3B MoE (Q4) | ~20 GB | Fits, only 3B active | `ollama run qwen3.5:35b-a3b` |

**Do NOT attempt locally:**
- Qwen3-VL-72B, Qwen3.5-397B — way too large
- Any full-precision (FP16) model above 14B

**Training Qwen:** Not feasible locally. Use cloud GPUs (Colab Pro A100, or cloud instances).

### YOLO Models

| Task | Feasibility | Notes |
|------|-------------|-------|
| Inference (any size) | Excellent, real-time | Uses MPS (Metal) backend automatically |
| Training nano/small | Good | Custom datasets up to ~10K images |
| Training medium | OK | Slower but works |
| Training large/xlarge | Slow | Works but consider Colab for speed |

YOLO uses MPS (Metal Performance Shaders) on Apple Silicon automatically via PyTorch.

## Setup

### Prerequisites

```bash
# Ollama for LLM/VLM inference
brew install ollama
ollama serve  # starts the server

# Python env for YOLO
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics torch torchvision
```

### Pull Models

```bash
# Pick the model size that fits your use case
ollama pull qwen3-vl:8b      # best balance for local dev
ollama pull qwen3.5:9b        # if you want built-in multimodal

# YOLO — no separate download, ultralytics auto-downloads
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## When to Use Cloud (Colab)

| Scenario | Recommendation |
|----------|---------------|
| YOLO inference | Local |
| YOLO training (small dataset <5K images) | Local |
| YOLO training (large dataset >10K images) | Colab Pro (faster) |
| Qwen VL inference (<=8B) | Local |
| Qwen VL inference (32B+) | Cloud or accept slow local |
| Qwen fine-tuning / LoRA | Cloud only (A100 40/80GB) |

## Alternative Local Runtimes

- **MLX** (Apple-optimized): Best performance on M-series for LLMs. `pip install mlx-lm`
- **llama.cpp**: C++ inference, very efficient. Ollama uses this under the hood.
- **vLLM**: Does NOT support MPS well — skip for local Mac use.
