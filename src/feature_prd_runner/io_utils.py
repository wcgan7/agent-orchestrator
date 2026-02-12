"""Read/write durable state files (JSON/YAML) and provide cross-platform file locking."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from .utils import _now_iso, _parse_iso

WINDOWS_LOCK_BYTES = 4096


def _require_yaml() -> None:
    if not yaml:
        raise RuntimeError("PyYAML is required to read/write .yaml files. Install pyyaml.")


class FileLock:
    """Provide a best-effort cross-platform file lock.

    This lock is advisory on platforms where that is the norm. It is intended
    to prevent concurrent runner processes from clobbering durable state files.

    Args:
        lock_path: Path to the lock file used as the lock target.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.handle: Optional[Any] = None
        self.lock_bytes = WINDOWS_LOCK_BYTES

    def __enter__(self) -> "FileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.lock_path, "w")
        try:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        except ImportError:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                self.handle.truncate(self.lock_bytes)
                self.handle.flush()
                locking = getattr(msvcrt, "locking", None)
                lk_lock = getattr(msvcrt, "LK_LOCK", None)
                if callable(locking) and lk_lock is not None:
                    locking(self.handle.fileno(), lk_lock, self.lock_bytes)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self.handle:
            return
        try:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        except ImportError:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                locking = getattr(msvcrt, "locking", None)
                lk_unlock = getattr(msvcrt, "LK_UNLCK", None)
                if callable(locking) and lk_unlock is not None:
                    locking(self.handle.fileno(), lk_unlock, self.lock_bytes)
        self.handle.close()
        self.handle = None


def _load_data(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        with open(path, "r") as handle:
            if path.suffix in {".yaml", ".yml"}:
                _require_yaml()
                data = yaml.safe_load(handle)
            else:
                data = json.load(handle)
        return data if isinstance(data, dict) else default
    except (OSError, json.JSONDecodeError):
        return default
    except Exception as exc:
        if yaml and isinstance(exc, yaml.YAMLError):
            return default
        raise


def _read_log_tail(path: Path, max_chars: int = 4000) -> str:
    return _read_text_tail(path, max_chars=max_chars)


def _read_text_tail(path: Path, *, max_chars: int = 4000, encoding: str = "utf-8") -> str:
    """Efficiently read the last ~`max_chars` characters of a text file."""
    if max_chars <= 0:
        return ""
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            # Overshoot in bytes to account for multi-byte encodings.
            max_bytes = min(size, max_chars * 4)
            handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
    except OSError:
        return ""
    text = data.decode(encoding, errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _heartbeat_from_progress(
    progress_path: Path,
    expected_run_id: Optional[str] = None,
) -> Optional[datetime]:
    if not progress_path.exists():
        return None
    progress = _load_data(progress_path, {})
    if expected_run_id:
        run_id = progress.get("run_id")
        if run_id and run_id != expected_run_id:
            return None
    heartbeat = _parse_iso(progress.get("heartbeat")) or _parse_iso(progress.get("timestamp"))
    if heartbeat:
        return heartbeat
    try:
        mtime = progress_path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except OSError:
        return None
