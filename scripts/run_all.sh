#!/bin/bash
# Master runner with checkpointing. Re-run safely — skips completed steps.
# Usage: ./scripts/run_all.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_DIR/.venv"
CHECKPOINT_DIR="$PROJECT_DIR/.checkpoints"

mkdir -p "$CHECKPOINT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
done_flag() { touch "$CHECKPOINT_DIR/$1.done"; }
is_done() { [ -f "$CHECKPOINT_DIR/$1.done" ]; }

activate_venv() {
  if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
  else
    log "ERROR: venv not found at $VENV. Run: python3 -m venv .venv && source .venv/bin/activate && pip install torch ultralytics opencv-python requests tqdm pillow"
    exit 1
  fi
}

# ============================================================
# STEP 1: Extract frames
# ============================================================
if is_done "step1_extract"; then
  log "SKIP: Frames already extracted"
else
  log "STEP 1: Extracting frames from videos..."
  activate_venv
  python3 "$SCRIPT_DIR/01_extract_frames.py"
  done_flag "step1_extract"
  log "DONE: Frame extraction"
fi

# ============================================================
# STEP 2: Auto-label with YOLO-World
# ============================================================
if is_done "step2_label"; then
  log "SKIP: Frames already labeled"
else
  log "STEP 2: Auto-labeling with YOLO-World..."
  activate_venv
  python3 "$SCRIPT_DIR/02_autolabel_yolo_world.py"
  done_flag "step2_label"
  log "DONE: Auto-labeling"
fi

# ============================================================
# STEP 3: Train YOLOv8n
# ============================================================
if is_done "step3_train"; then
  log "SKIP: YOLOv8n already trained"
else
  log "STEP 3: Training YOLOv8n (this takes ~20-40min on M1 Pro)..."
  activate_venv
  python3 "$SCRIPT_DIR/03_train_yolo.py"
  done_flag "step3_train"
  log "DONE: YOLOv8n training"
fi

# ============================================================
# STEP 4: Check ollama models
# ============================================================
log "STEP 4: Checking ollama models..."
if ! ollama list | grep -q "qwen3-vl:8b"; then
  log "Pulling qwen3-vl:8b (6.1GB)..."
  ollama pull qwen3-vl:8b
fi
if ! ollama list | grep -q "qwen3.5:35b"; then
  log "Pulling qwen3.5:35b (24GB)..."
  ollama pull qwen3.5:35b
fi
log "Models ready"

# ============================================================
# STEP 5: Benchmark VLMs
# ============================================================
if is_done "step5_benchmark"; then
  log "SKIP: VLM benchmarks already done"
else
  log "STEP 5: Benchmarking VLMs (this takes ~20-60min for 20 frames each)..."
  activate_venv
  python3 "$SCRIPT_DIR/04_benchmark_vlm.py"
  done_flag "step5_benchmark"
  log "DONE: VLM benchmarks"
fi

# ============================================================
# STEP 6: Compile results
# ============================================================
log "STEP 6: Compiling results into docs/benchmark-results.md..."
activate_venv
python3 "$SCRIPT_DIR/05_compile_results.py"
log "DONE: Results written to docs/benchmark-results.md"

log "=== ALL DONE ==="
log "Results: $PROJECT_DIR/docs/benchmark-results.md"
log "Raw data: $PROJECT_DIR/docs/raw/"
