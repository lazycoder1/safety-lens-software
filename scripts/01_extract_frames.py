#!/usr/bin/env python3
"""Extract frames from test videos for YOLO dataset creation."""
import cv2
import os
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent / "test-videos"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "frames"

FRAMES_PER_VIDEO = 15  # extract 15 evenly-spaced frames per video


def extract_frames(video_path: Path, output_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0

    print(f"  {video_path.name}: {total} frames, {fps:.1f}fps, {duration:.1f}s")

    if total == 0:
        cap.release()
        return []

    step = max(1, total // FRAMES_PER_VIDEO)
    saved = []
    frame_idx = 0
    extracted = 0

    while extracted < FRAMES_PER_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        stem = video_path.stem
        out_path = output_dir / f"{stem}_f{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(out_path)
        extracted += 1
        frame_idx += step

    cap.release()
    return saved


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} videos. Extracting {FRAMES_PER_VIDEO} frames each...")
    all_frames = []

    for video in videos:
        frames = extract_frames(video, OUTPUT_DIR)
        all_frames.extend(frames)
        print(f"  -> saved {len(frames)} frames")

    print(f"\nTotal frames extracted: {len(all_frames)}")
    print(f"Saved to: {OUTPUT_DIR}")
    return all_frames


if __name__ == "__main__":
    main()
