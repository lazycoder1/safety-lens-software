"""Paths and crash-safe writes for mutable edge runtime state."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


BACKEND_DIR = Path(__file__).parent


def resolve_state_dir() -> Path:
    configured = os.environ.get("SAFETYLENS_STATE_DIR", "").strip()
    if not configured:
        return BACKEND_DIR
    state_dir = Path(configured).expanduser()
    if not state_dir.is_absolute():
        raise RuntimeError("SAFETYLENS_STATE_DIR must be an absolute path")
    return state_dir


STATE_DIR = resolve_state_dir()


def atomic_write_file(
    path: Path,
    content: bytes,
    *,
    file_mode: int = 0o600,
    directory_mode: int | None = None,
) -> None:
    """Atomically replace a file, flush it, then flush its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=directory_mode or 0o755)
    if directory_mode is not None:
        os.chmod(path.parent, directory_mode)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, file_mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, file_mode)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_private(path: Path, content: bytes) -> None:
    """Persist private state with durable 0700/0600 permissions."""
    atomic_write_file(path, content, file_mode=0o600, directory_mode=0o700)
