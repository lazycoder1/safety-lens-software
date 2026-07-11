"""
Structured logging for Rakshak Lens.

Dev  → colorized console (human-readable) + rotating file
Prod → JSON-line console (for journalctl) + rotating file

Usage:
    from logging_config import setup_logging
    setup_logging()

    import logging
    logger = logging.getLogger("rakshak_lens.video")
    logger.info("Camera started", extra={"camera_id": "cam1"})

Env vars:
    SAFETYLENS_ENV      dev | prod  (default: dev)
    SAFETYLENS_LOG_LEVEL  DEBUG | INFO | WARNING | ERROR  (optional override)
"""

import json
import logging
import os
import socket
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

SERVICE_NAME = "rakshak_lens"
HOSTNAME = socket.gethostname()

DEFAULT_LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR = Path(os.environ.get("SAFETYLENS_LOG_DIR", str(DEFAULT_LOGS_DIR)))
DEFAULT_LOG_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 8


class JSONFormatter(logging.Formatter):
    """One JSON object per line — parseable by jq, journalctl, ELK."""

    _env = os.environ.get("SAFETYLENS_ENV", "dev").lower()

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "environment": self._env,
            "host": HOSTNAME,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra keys (camera_id, path, etc.)
        for key in (
            "camera_id",
            "path",
            "source",
            "subscribers",
            "elapsed",
            "alert_id",
            "request_id",
            "method",
            "route",
            "status_code",
            "duration_ms",
            "user_id",
            "username",
            "action",
            "target_type",
            "target_id",
            "outcome",
            "free_bytes",
        ):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


class ColorConsoleFormatter(logging.Formatter):
    """Human-readable with ANSI colors for dev."""

    COLORS = {
        "DEBUG": "\033[90m",     # gray
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red bg
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        name_short = record.name.replace("rakshak_lens.", "")
        base = f"{color}[{ts}] {record.levelname:<8}{self.RESET} {name_short:<12} {record.getMessage()}"

        # Append extra context
        extras = []
        for key in (
            "camera_id",
            "path",
            "source",
            "elapsed",
            "request_id",
            "status_code",
            "duration_ms",
            "route",
            "action",
        ):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        if extras:
            base += f"  ({', '.join(extras)})"

        if record.exc_info and record.exc_info[0] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging() -> None:
    """Call once at application startup."""
    env = os.environ.get("SAFETYLENS_ENV", "dev").lower()
    level_override = os.environ.get("SAFETYLENS_LOG_LEVEL", "").upper()
    is_prod = env == "prod"

    root_logger = logging.getLogger("rakshak_lens")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    # ── Console handler ──────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    if is_prod:
        console.setFormatter(JSONFormatter())
    else:
        console.setFormatter(ColorConsoleFormatter())
    root_logger.addHandler(console)

    # ── File handlers ──────────────────────────────────────────────────
    LOGS_DIR.mkdir(exist_ok=True)
    max_bytes = int(os.environ.get("SAFETYLENS_LOG_MAX_BYTES", str(DEFAULT_LOG_MAX_BYTES)))
    backup_count = int(os.environ.get("SAFETYLENS_LOG_BACKUP_COUNT", str(DEFAULT_LOG_BACKUP_COUNT)))

    # Main log — INFO and above (requests, state changes, startup)
    file_handler = RotatingFileHandler(
        LOGS_DIR / "rakshak-lens.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Error log — WARNING and above only (the "something is wrong" log)
    error_file_handler = RotatingFileHandler(
        LOGS_DIR / "errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.WARNING)
    error_file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_file_handler)

    # ── Optional level override ──────────────────────────────────────
    if level_override and hasattr(logging, level_override):
        console.setLevel(getattr(logging, level_override))

    # Quiet noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    root_logger.info(
        "Logging initialized",
        extra={
            "source": f"env={env}",
            "path": str(LOGS_DIR),
        },
    )
