#!/usr/bin/env python3
"""Train YOLOv8n on the auto-labeled industrial safety dataset."""
import time
import json
from pathlib import Path

from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent.parent / "dataset"
RESULTS_DIR = Path(__file__).parent.parent / "docs" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
IMG_SIZE = 640
BATCH = 8  # conservative for M1 Pro 32GB


def main():
    yaml_path = DATASET_DIR / "dataset.yaml"
    if not yaml_path.exists():
        print(f"Dataset YAML not found: {yaml_path}")
        print("Run 02_autolabel_yolo_world.py first.")
        return

    print("=== Training YOLOv8n ===")
    print(f"Dataset: {yaml_path}")
    print(f"Epochs: {EPOCHS}, img_size: {IMG_SIZE}, batch: {BATCH}")
    print()

    model = YOLO("yolov8n.pt")

    start = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="mps",
        project=str(Path(__file__).parent.parent / "runs"),
        name="industrial_safety_yolov8n",
        verbose=True,
        plots=True,
    )
    elapsed = time.time() - start

    # Extract metrics
    metrics = model.val()

    # Benchmark inference speed on val set
    val_images = list((DATASET_DIR / "images" / "val").glob("*.jpg"))[:20]

    inf_start = time.time()
    for img in val_images:
        model.predict(str(img), verbose=False)
    inf_elapsed = time.time() - inf_start
    fps = len(val_images) / inf_elapsed if inf_elapsed > 0 else 0

    output = {
        "model": "yolov8n",
        "training": {
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "batch": BATCH,
            "device": "mps (Apple M1 Pro)",
            "training_time_seconds": round(elapsed, 1),
            "training_time_minutes": round(elapsed / 60, 1),
        },
        "metrics": {
            "mAP50": round(float(metrics.box.map50), 4) if hasattr(metrics, 'box') else None,
            "mAP50_95": round(float(metrics.box.map), 4) if hasattr(metrics, 'box') else None,
            "precision": round(float(metrics.box.mp), 4) if hasattr(metrics, 'box') else None,
            "recall": round(float(metrics.box.mr), 4) if hasattr(metrics, 'box') else None,
        },
        "inference": {
            "frames_tested": len(val_images),
            "total_time_seconds": round(inf_elapsed, 3),
            "fps": round(fps, 1),
            "ms_per_frame": round(1000 / fps, 1) if fps > 0 else None,
        },
    }

    out_path = RESULTS_DIR / "yolo_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n=== YOLOv8n Training Results ===")
    print(json.dumps(output, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
