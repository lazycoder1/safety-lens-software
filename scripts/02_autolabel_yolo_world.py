#!/usr/bin/env python3
"""
Auto-label extracted frames using YOLO-World zero-shot detection.
Generates YOLO-format annotations for industrial safety classes.
"""
import os
import sys
import shutil
import random
from pathlib import Path

from ultralytics import YOLOWorld

DATASET_DIR = Path(__file__).parent.parent / "dataset"
FRAMES_DIR = DATASET_DIR / "frames"

# Industrial safety classes to detect
CLASSES = [
    "person",
    "worker",
    "hard hat",
    "helmet",
    "safety vest",
    "welding equipment",
    "machinery",
    "electrical equipment",
    "power tools",
]

# Merge similar classes for YOLO training
CLASS_MAP = {
    "person": 0,
    "worker": 0,       # same as person
    "hard hat": 1,
    "helmet": 1,       # same as hard hat
    "safety vest": 2,
    "welding equipment": 3,
    "machinery": 4,
    "electrical equipment": 4,  # merge with machinery
    "power tools": 4,
}

FINAL_CLASSES = ["person", "hard_hat", "safety_vest", "equipment"]

CONFIDENCE_THRESHOLD = 0.25
TRAIN_SPLIT = 0.8


def setup_dirs():
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def label_frames(model, frames: list[Path]) -> dict:
    stats = {"total": len(frames), "labeled": 0, "detections": 0, "skipped": 0}

    for frame_path in frames:
        results = model.predict(str(frame_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            stats["skipped"] += 1
            continue

        h, w = result.orig_shape
        lines = []

        for box in boxes:
            cls_name = result.names[int(box.cls)].lower()
            # Find matching class
            mapped_id = None
            for key, val in CLASS_MAP.items():
                if key in cls_name or cls_name in key:
                    mapped_id = val
                    break
            if mapped_id is None:
                continue

            # Convert to YOLO format (normalized xywh)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{mapped_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            stats["labeled"] += 1
            stats["detections"] += len(lines)
            yield frame_path, lines
        else:
            stats["skipped"] += 1

    return stats


def main():
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        print(f"No frames found in {FRAMES_DIR}. Run 01_extract_frames.py first.")
        sys.exit(1)

    print(f"Found {len(frames)} frames to label")
    setup_dirs()

    # Load YOLO-World
    print("Loading YOLO-World model...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(CLASSES)

    # Split frames
    random.seed(42)
    random.shuffle(frames)
    split_idx = int(len(frames) * TRAIN_SPLIT)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    total_labeled = 0
    total_dets = 0
    skipped = 0

    for split, frame_set in [("train", train_frames), ("val", val_frames)]:
        print(f"\nLabeling {split} set ({len(frame_set)} frames)...")
        for frame_path, lines in label_frames(model, frame_set):
            # Copy image
            dest_img = DATASET_DIR / "images" / split / frame_path.name
            shutil.copy2(frame_path, dest_img)

            # Write label
            label_path = DATASET_DIR / "labels" / split / (frame_path.stem + ".txt")
            label_path.write_text("\n".join(lines))

            total_labeled += 1
            total_dets += len(lines)
            print(f"  {frame_path.name}: {len(lines)} detections")

        skipped += len(frame_set) - sum(
            1 for f in frame_set
            if (DATASET_DIR / "labels" / split / (f.stem + ".txt")).exists()
        )

    # Write dataset YAML
    yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

nc: {len(FINAL_CLASSES)}
names: {FINAL_CLASSES}
"""
    (DATASET_DIR / "dataset.yaml").write_text(yaml_content)

    print(f"\n=== Auto-labeling complete ===")
    print(f"Frames labeled: {total_labeled}/{len(frames)}")
    print(f"Total detections: {total_dets}")
    print(f"Skipped (no detections): {skipped}")
    print(f"Dataset YAML: {DATASET_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
