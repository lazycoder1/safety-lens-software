"""
Rakshak Lens constants — paths, class maps, color palettes, public routes.
"""

from pathlib import Path
import math
import os


def _finite_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a finite, bounded float without allowing invalid startup state."""
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(maximum, max(minimum, value))


# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "coco_primary" / "yolo26n.pt"
YOLOE_MODEL_PATH = PROJECT_ROOT / "models" / "yoloe_open_vocab" / "yoloe-26s-seg.pt"
VIDEO_DIR = PROJECT_ROOT / "test-videos"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
try:
    VLM_TIMEOUT_SECONDS = float(os.environ.get("SAFETYLENS_VLM_TIMEOUT_SECONDS", "60"))
except (TypeError, ValueError):
    VLM_TIMEOUT_SECONDS = 60.0
VLM_TIMEOUT_SECONDS = min(300.0, max(5.0, VLM_TIMEOUT_SECONDS))
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"
MODEL_SERVER_URL = os.environ.get("SAFETYLENS_MODEL_SERVER_URL", "").rstrip("/")
MODEL_SERVER_TOKEN = os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", "")
MODEL_SERVER_TIMEOUT_SECONDS = _finite_env_float(
    "SAFETYLENS_MODEL_SERVER_TIMEOUT_SECONDS",
    30.0,
    minimum=0.1,
    maximum=300.0,
)


# Model catalogue calls sit on camera-start and health paths, so they use a
# short, dedicated budget instead of the much larger inference timeout.
MODEL_METADATA_TIMEOUT_SECONDS = _finite_env_float(
    "SAFETYLENS_MODEL_METADATA_TIMEOUT_SECONDS",
    2.0,
    minimum=0.1,
    maximum=30.0,
)
MODEL_METADATA_TTL_SECONDS = _finite_env_float(
    "SAFETYLENS_MODEL_METADATA_TTL_SECONDS",
    5.0,
    minimum=0.1,
    maximum=300.0,
)
MODEL_METADATA_BREAKER_INITIAL_SECONDS = _finite_env_float(
    "SAFETYLENS_MODEL_METADATA_BREAKER_INITIAL_SECONDS",
    1.0,
    minimum=0.1,
    maximum=60.0,
)
MODEL_METADATA_BREAKER_MAX_SECONDS = max(
    MODEL_METADATA_BREAKER_INITIAL_SECONDS,
    _finite_env_float(
        "SAFETYLENS_MODEL_METADATA_BREAKER_MAX_SECONDS",
        10.0,
        minimum=0.1,
        maximum=300.0,
    ),
)
MODEL_METADATA_LOG_REMINDER_SECONDS = _finite_env_float(
    "SAFETYLENS_MODEL_METADATA_LOG_REMINDER_SECONDS",
    60.0,
    minimum=1.0,
    maximum=86_400.0,
)

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

PUBLIC_PATHS = {"/api/auth/login", "/api/auth/register", "/api/health", "/api/ping", "/docs", "/openapi.json"}
PUBLIC_PREFIXES = ("/api/stream/", "/api/snapshots/")

# ── Violation detection threshold ───────────────────────────────────────────

VIOLATION_THRESHOLD = 10  # must persist for N consecutive detection frames before firing (~5 seconds at 6fps/3rd frame)
