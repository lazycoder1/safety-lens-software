"""
SafetyLens constants — paths, class maps, color palettes, public routes.
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"  # COCO pretrained YOLO26 — 80 classes, NMS-free
YOLOE_MODEL_PATH = PROJECT_ROOT / "yoloe-11s-seg.pt"
VIDEO_DIR = PROJECT_ROOT / "test-videos"
OLLAMA_URL = "http://localhost:11434/api/generate"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

# ── COCO class names (80 classes) ───────────────────────────────────────────

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# ── Safety-relevant COCO classes with colors ────────────────────────────────

SAFETY_CLASSES = {0, 15, 16, 67, 7, 2, 3}  # person, cat, dog, cell_phone, truck, car, motorcycle
CLASS_COLORS = {
    0: (59, 130, 246),   # person - blue
    15: (234, 179, 8),   # cat - yellow
    16: (234, 179, 8),   # dog - yellow
    67: (239, 68, 68),   # cell phone - red
    7: (168, 85, 247),   # truck - purple
    2: (168, 162, 158),  # car - gray
    3: (249, 115, 22),   # motorcycle - orange
}

# ── Color palette for YOLOe open-vocabulary classes ─────────────────────────

YOLOE_COLORS = [
    (59, 130, 246),   # blue
    (34, 197, 94),    # green
    (234, 179, 8),    # yellow
    (239, 68, 68),    # red
    (168, 85, 247),   # purple
    (236, 72, 153),   # pink
    (20, 184, 166),   # teal
    (249, 115, 22),   # orange
    (99, 102, 241),   # indigo
    (6, 182, 212),    # cyan
]

# ── Auth middleware paths ───────────────────────────────────────────────────

PUBLIC_PATHS = {"/api/auth/login", "/api/auth/register", "/api/health", "/docs", "/openapi.json"}
PUBLIC_PREFIXES = ("/api/stream/", "/api/snapshots/")

# ── Violation detection threshold ───────────────────────────────────────────

VIOLATION_THRESHOLD = 10  # must persist for N consecutive detection frames before firing (~5 seconds at 6fps/3rd frame)
