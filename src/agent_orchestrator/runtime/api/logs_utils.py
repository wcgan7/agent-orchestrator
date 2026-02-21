"""Log and state-file helper utilities for runtime API routes."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Optional


def _safe_state_path(raw_path: Any, state_root: Path) -> Optional[Path]:
    if not raw_path:
        return None
    try:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = state_root / path
        resolved = path.resolve()
        state_resolved = state_root.resolve()
        if resolved == state_resolved or state_resolved in resolved.parents:
            return resolved
        return None
    except Exception:
        return None


def _read_tail(path: Optional[Path], max_chars: int) -> tuple[str, int]:
    """Read the tail of a file.

    Returns ``(text, tail_start_byte)`` where *tail_start_byte* is the byte
    offset in the file where the returned text begins.  When the entire file
    fits within *max_chars*, ``tail_start_byte`` is ``0``.
    """
    if not path or not path.exists() or not path.is_file() or max_chars <= 0:
        return "", 0
    try:
        size = path.stat().st_size
    except Exception:
        return "", 0
    if size <= 0:
        return "", 0

    # Read only a bounded tail window (overshoot bytes for multibyte chars).
    max_bytes = min(size, max_chars * 4)
    start_byte = max(0, size - max_bytes)
    prev_byte = b"\n"
    try:
        with open(path, "rb") as fh:
            if start_byte > 0:
                fh.seek(start_byte - 1)
                prev_byte = fh.read(1) or b"\n"
            fh.seek(start_byte)
            raw = fh.read(max_bytes)
    except Exception:
        return "", 0

    text = raw.decode("utf-8", errors="replace")
    if not text:
        return "", size

    cut_at = max(0, len(text) - max_chars)
    slice_starts_mid_line = False
    if cut_at > 0:
        slice_starts_mid_line = text[cut_at - 1] != "\n"
    elif start_byte > 0 and prev_byte != b"\n":
        slice_starts_mid_line = True

    # If the returned slice starts in the middle of a line, drop the orphaned
    # prefix so NDJSON/text consumers only see full lines.
    if slice_starts_mid_line:
        first_newline = text.find("\n", cut_at)
        if first_newline == -1:
            return "", size
        cut_at = first_newline + 1

    tail = text[cut_at:]
    tail_start_byte = start_byte + len(text[:cut_at].encode("utf-8"))
    return tail, tail_start_byte


def _read_from_offset(
    path: Optional[Path],
    offset: int,
    max_bytes: int,
    read_to: int = 0,
    *,
    align_start_to_line: bool = False,
) -> tuple[str, int, int]:
    """Read file content from *offset* bytes forward.

    Returns ``(text, new_offset, chunk_start_offset)`` where *new_offset* is the
    byte offset after the returned chunk and *chunk_start_offset* is the actual
    byte offset where the returned chunk begins.

    If *read_to* > 0, do not read past that byte position in the file.
    """
    if not path or not path.exists() or not path.is_file():
        return "", 0, 0
    try:
        size = path.stat().st_size
        end = min(size, read_to) if read_to > 0 else size
        start = max(offset, 0)
        if start >= end:
            bounded = min(start, size)
            return "", bounded, bounded

        aligned_start = start
        if align_start_to_line and start > 0:
            lookback = min(start, max(max_bytes, 8192))
            with open(path, "rb") as fh:
                fh.seek(start - lookback)
                prefix = fh.read(lookback)
            newline_idx = prefix.rfind(b"\n")
            if newline_idx >= 0:
                aligned_start = (start - lookback) + newline_idx + 1

        read_limit = end - aligned_start
        if read_limit <= 0:
            bounded = min(aligned_start, size)
            return "", bounded, bounded
        # Allow aligned reads to exceed ``max_bytes`` by at most one chunk size.
        capped = min(read_limit, max_bytes * 2 if align_start_to_line else max_bytes)
        with open(path, "rb") as fh:
            fh.seek(aligned_start)
            raw = fh.read(capped)
        text = raw.decode("utf-8", errors="replace")
        return text, aligned_start + len(raw), aligned_start
    except Exception:
        return "", 0, 0


def _logs_snapshot_id(logs_meta: dict[str, Any]) -> str:
    parts = [
        str(logs_meta.get("run_dir") or ""),
        str(logs_meta.get("stdout_path") or ""),
        str(logs_meta.get("stderr_path") or ""),
        str(logs_meta.get("started_at") or ""),
    ]
    material = "|".join(parts).strip("|")
    if not material:
        return ""
    return sha256(material.encode("utf-8", errors="ignore")).hexdigest()[:16]
