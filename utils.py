from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if not isinstance(value, str):
            value = str(value)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        # If a naive timestamp slips in, assume UTC to avoid crashes.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def _is_placeholder_text(value: str) -> bool:
    normalized = _normalize_text(value)
    normalized = normalized.strip().strip("()[]{}").strip(".,;:")
    return normalized in {"none", "n/a", "na", "nil", "null"}


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _validate_string_list(
    value: Any,
    field_name: str,
    min_items: int = 1,
    max_items: Optional[int] = None,
) -> tuple[bool, str]:
    if not isinstance(value, list) or len(value) < min_items:
        return False, f"{field_name} must be a list with at least {min_items} items"
    if max_items is not None and len(value) > max_items:
        return False, f"{field_name} must have no more than {max_items} items"
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return False, f"{field_name} must contain non-empty strings"
    return True, ""


def _sanitize_phase_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "phase"


def _hash_json_data(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pid_is_running(pid_value: Any) -> bool:
    pid = _coerce_int(pid_value, 0)
    if pid <= 0:
        return False

    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except PermissionError:
            # Process exists but we may not have permission to signal it.
            return True
        except OSError:
            return False
        else:
            return True

    # Windows: best-effort OpenProcess check.
    try:
        import ctypes
        import ctypes.wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            ctypes.wintypes.DWORD(pid),
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        # If we can't determine, treat as NOT running to avoid deadlocks.
        # (Worker PID check + heartbeat still provide safety.)
        return False
