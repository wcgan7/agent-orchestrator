"""Helpers for one-shot human guidance injection across retry/change flows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
import uuid

from ..domain.models import Task, now_iso

ACTIVE_HUMAN_GUIDANCE_KEY = "active_human_guidance"
ACTIVE_HUMAN_GUIDANCE_CLEARED_AT_KEY = "active_human_guidance_cleared_at"

_GUIDANCE_SOURCE_SET = {"gate_request_changes", "review_request_changes", "retry"}


def _ensure_metadata(task: Task) -> dict[str, Any]:
    if not isinstance(task.metadata, dict):
        task.metadata = {}
    return task.metadata


def _parse_iso_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_step(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_guidance_envelope(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    guidance = str(value.get("guidance") or "").strip()
    if not guidance:
        return None
    source = str(value.get("source") or "retry").strip().lower()
    if source not in _GUIDANCE_SOURCE_SET:
        source = "retry"
    envelope: dict[str, Any] = {
        "id": str(value.get("id") or f"hg-{uuid.uuid4().hex[:10]}"),
        "source": source,
        "guidance": guidance,
        "created_at": str(value.get("created_at") or now_iso()),
        "target_step": _normalize_step(value.get("target_step")),
        "fallback_step": _normalize_step(value.get("fallback_step")),
        "gate": _normalize_step(value.get("gate")),
        "consumed": bool(value.get("consumed")),
        "consumed_at": _normalize_step(value.get("consumed_at")),
        "consumed_step": _normalize_step(value.get("consumed_step")),
        "consumed_run_id": _normalize_step(value.get("consumed_run_id")),
    }
    return envelope


def _set_envelope(task: Task, envelope: dict[str, Any]) -> dict[str, Any]:
    metadata = _ensure_metadata(task)
    metadata[ACTIVE_HUMAN_GUIDANCE_KEY] = envelope
    metadata.pop(ACTIVE_HUMAN_GUIDANCE_CLEARED_AT_KEY, None)
    return envelope


def set_active_human_guidance(
    task: Task,
    *,
    source: str,
    guidance: str,
    created_at: str | None = None,
    target_step: str | None = None,
    fallback_step: str | None = None,
    gate: str | None = None,
) -> dict[str, Any] | None:
    """Set active guidance envelope, replacing any previous one."""
    normalized_guidance = str(guidance or "").strip()
    if not normalized_guidance:
        return None
    normalized_source = str(source or "").strip().lower()
    if normalized_source not in _GUIDANCE_SOURCE_SET:
        normalized_source = "retry"
    envelope = {
        "id": f"hg-{uuid.uuid4().hex[:10]}",
        "source": normalized_source,
        "guidance": normalized_guidance,
        "created_at": str(created_at or now_iso()),
        "target_step": _normalize_step(target_step),
        "fallback_step": _normalize_step(fallback_step),
        "gate": _normalize_step(gate),
        "consumed": False,
        "consumed_at": None,
        "consumed_step": None,
        "consumed_run_id": None,
    }
    return _set_envelope(task, envelope)


def clear_active_human_guidance(task: Task, *, cleared_at: str | None = None) -> None:
    """Clear active guidance and set a tombstone to suppress stale legacy promotion."""
    metadata = _ensure_metadata(task)
    metadata.pop(ACTIVE_HUMAN_GUIDANCE_KEY, None)
    metadata[ACTIVE_HUMAN_GUIDANCE_CLEARED_AT_KEY] = str(cleared_at or now_iso())


def _legacy_candidates(task: Task) -> list[dict[str, Any]]:
    metadata = _ensure_metadata(task)
    candidates: list[dict[str, Any]] = []

    requested_changes = metadata.get("requested_changes")
    if isinstance(requested_changes, dict):
        guidance = str(requested_changes.get("guidance") or "").strip()
        if guidance:
            candidates.append(
                {
                    "source": "gate_request_changes" if str(requested_changes.get("gate") or "").strip() else "review_request_changes",
                    "guidance": guidance,
                    "created_at": str(requested_changes.get("ts") or now_iso()),
                    "target_step": _normalize_step(metadata.get("retry_from_step")),
                    "fallback_step": "implement_fix",
                    "gate": _normalize_step(requested_changes.get("gate")),
                    "_origin_index": 0,
                }
            )

    retry_guidance = metadata.get("retry_guidance")
    if isinstance(retry_guidance, dict):
        guidance = str(retry_guidance.get("guidance") or "").strip()
        if guidance:
            candidates.append(
                {
                    "source": "retry",
                    "guidance": guidance,
                    "created_at": str(retry_guidance.get("ts") or now_iso()),
                    "target_step": _normalize_step(metadata.get("retry_from_step")),
                    "fallback_step": "implement_fix",
                    "gate": None,
                    "_origin_index": 1,
                }
            )
    return candidates


def _filter_legacy_by_clear_marker(candidates: list[dict[str, Any]], cleared_at: Any) -> list[dict[str, Any]]:
    cleared_ts = _parse_iso_timestamp(cleared_at)
    if cleared_ts is None:
        return candidates
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        created_ts = _parse_iso_timestamp(candidate.get("created_at"))
        if created_ts is None:
            continue
        if created_ts <= cleared_ts:
            continue
        filtered.append(candidate)
    return filtered


def promote_legacy_human_guidance(task: Task) -> bool:
    """Promote legacy requested/retry guidance into active envelope when missing."""
    metadata = _ensure_metadata(task)
    existing = _normalize_guidance_envelope(metadata.get(ACTIVE_HUMAN_GUIDANCE_KEY))
    if existing is not None:
        metadata[ACTIVE_HUMAN_GUIDANCE_KEY] = existing
        return False

    candidates = _legacy_candidates(task)
    if not candidates:
        return False
    candidates = _filter_legacy_by_clear_marker(candidates, metadata.get(ACTIVE_HUMAN_GUIDANCE_CLEARED_AT_KEY))
    if not candidates:
        return False

    def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, int]:
        ts = _parse_iso_timestamp(candidate.get("created_at"))
        score = ts.timestamp() if ts else 0.0
        origin_index = int(candidate.get("_origin_index") or 0)
        return (score, -origin_index)

    selected = max(candidates, key=_candidate_rank)
    selected.pop("_origin_index", None)
    _set_envelope(task, _normalize_guidance_envelope(selected) or selected)
    return True


def active_human_guidance(task: Task, *, promote_legacy: bool = True) -> dict[str, Any] | None:
    """Return normalized active guidance envelope for a task."""
    metadata = _ensure_metadata(task)
    envelope = _normalize_guidance_envelope(metadata.get(ACTIVE_HUMAN_GUIDANCE_KEY))
    if envelope is not None:
        metadata[ACTIVE_HUMAN_GUIDANCE_KEY] = envelope
        return envelope
    if promote_legacy and promote_legacy_human_guidance(task):
        promoted = _normalize_guidance_envelope(metadata.get(ACTIVE_HUMAN_GUIDANCE_KEY))
        if promoted is not None:
            metadata[ACTIVE_HUMAN_GUIDANCE_KEY] = promoted
            return promoted
    return None


def guidance_for_step(task: Task, step: str, *, promote_legacy: bool = True) -> dict[str, Any] | None:
    """Return active guidance only when it applies to the provided step."""
    envelope = active_human_guidance(task, promote_legacy=promote_legacy)
    if envelope is None:
        return None
    if bool(envelope.get("consumed")):
        return None
    step_name = _normalize_step(step)
    if step_name is None:
        return None
    target_step = _normalize_step(envelope.get("target_step"))
    fallback_step = _normalize_step(envelope.get("fallback_step"))
    if target_step and step_name == target_step:
        return envelope
    if fallback_step and step_name == fallback_step:
        return envelope
    if not target_step and not fallback_step:
        current_step = _normalize_step(getattr(task, "current_step", None))
        if current_step and current_step == step_name:
            return envelope
    return None


def render_human_guidance_prompt(task: Task, step: str, *, promote_legacy: bool = True) -> str | None:
    """Render user guidance block for a step prompt when guidance applies."""
    envelope = guidance_for_step(task, step, promote_legacy=promote_legacy)
    if envelope is None:
        return None
    guidance = str(envelope.get("guidance") or "").strip()
    if not guidance:
        return None
    source = str(envelope.get("source") or "").strip().lower()
    source_hint = {
        "gate_request_changes": "Source: approval gate request changes.",
        "review_request_changes": "Source: review request changes.",
        "retry": "Source: manual retry guidance.",
    }.get(source, "Source: user guidance.")
    gate = _normalize_step(envelope.get("gate"))
    lines = [source_hint]
    if gate:
        lines.append(f"Gate: {gate}")
    lines.append(guidance)
    return "\n".join(lines)


def consume_human_guidance_for_step(
    task: Task,
    *,
    step: str,
    run_id: str | None = None,
    consumed_at: str | None = None,
    promote_legacy: bool = True,
) -> bool:
    """Mark active guidance consumed if it applies to the completed step."""
    envelope = guidance_for_step(task, step, promote_legacy=promote_legacy)
    if envelope is None:
        return False
    envelope["consumed"] = True
    envelope["consumed_at"] = str(consumed_at or now_iso())
    envelope["consumed_step"] = _normalize_step(step)
    envelope["consumed_run_id"] = _normalize_step(run_id)
    _set_envelope(task, envelope)
    return True
