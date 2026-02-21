"""Compatibility exports for runtime API router and log I/O helpers."""

from __future__ import annotations

from .logs_io import read_from_offset, read_tail
from .router_impl import create_router


# Backwards-compatible helper names imported by tests and existing consumers.
_read_tail = read_tail
_read_from_offset = read_from_offset

__all__ = ["create_router", "_read_tail", "_read_from_offset"]
