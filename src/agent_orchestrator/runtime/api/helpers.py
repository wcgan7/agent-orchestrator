"""Helper utilities shared by runtime API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

IMPORT_JOB_TTL_SECONDS = 60 * 60 * 24
IMPORT_JOB_MAX_RECORDS = 200


def _priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 9)


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pruned_import_jobs(items: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    kept: list[dict[str, Any]] = []
    for item in items.values():
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("id") or "").strip()
        if not job_id:
            continue
        created_at = _parse_iso_datetime(item.get("created_at"))
        if created_at and (now - created_at).total_seconds() > IMPORT_JOB_TTL_SECONDS:
            continue
        kept.append(item)

    kept.sort(key=lambda job: str(job.get("created_at") or ""), reverse=True)
    trimmed = kept[:IMPORT_JOB_MAX_RECORDS]
    return {str(job.get("id")): job for job in trimmed if str(job.get("id") or "").strip()}


def _coerce_int(value: Any, default: int, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return default


def _normalize_str_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        k = str(key or "").strip()
        v = str(raw or "").strip()
        if k and v:
            out[k] = v
    return out


def _normalize_human_blocking_issues(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            summary = item.strip()
            if summary:
                out.append({"summary": summary})
            continue
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or item.get("issue") or "").strip()
        details = str(item.get("details") or item.get("rationale") or "").strip()
        if not summary and details:
            summary = details.splitlines()[0][:200].strip()
        if not summary:
            continue
        issue: dict[str, str] = {"summary": summary}
        if details:
            issue["details"] = details
        for key in ("category", "action", "blocking_on", "severity"):
            raw = item.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                issue[key] = text
        out.append(issue)
    return out[:20]
