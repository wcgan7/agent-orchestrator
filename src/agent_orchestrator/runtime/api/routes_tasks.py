"""Task-focused route registration for the runtime API."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

from fastapi import APIRouter, HTTPException, Query

from ...collaboration.modes import normalize_hitl_mode
from ...pipelines.registry import PipelineRegistry
from ..orchestrator.human_guidance import (
    clear_active_human_guidance,
    set_active_human_guidance,
)
from ..domain.models import (
    DependencyPolicy,
    Priority,
    Task,
    TaskStatus,
    now_iso,
)
from ..storage.bootstrap import (
    archive_state_root,
    archive_task_context_manifest,
    ensure_state_root,
)
from .deps import RouteDeps
from . import router_impl as impl


AddDependencyRequest = impl.AddDependencyRequest
ApproveGateRequest = impl.ApproveGateRequest
CommitPlanRequest = impl.CommitPlanRequest
CreatePlanRevisionRequest = impl.CreatePlanRevisionRequest
CreateTaskRequest = impl.CreateTaskRequest
GenerateTasksRequest = impl.GenerateTasksRequest
PipelineClassificationRequest = impl.PipelineClassificationRequest
PipelineClassificationResponse = impl.PipelineClassificationResponse
PlanRefineRequest = impl.PlanRefineRequest
RetryTaskRequest = impl.RetryTaskRequest
SkipToPrecommitRequest = impl.SkipToPrecommitRequest
TransitionRequest = impl.TransitionRequest
UpdateTaskRequest = impl.UpdateTaskRequest
VALID_TRANSITIONS = impl.VALID_TRANSITIONS
_canonical_task_type_for_pipeline = impl._canonical_task_type_for_pipeline
_execution_batches = impl._execution_batches
_gate_display_label = impl._gate_display_label
_has_unresolved_blockers = impl._has_unresolved_blockers
_logs_snapshot_id = impl._logs_snapshot_id
_normalize_pipeline_classification_output = impl._normalize_pipeline_classification_output
_priority_rank = impl._priority_rank
_read_from_offset = impl._read_from_offset
_read_tail = impl._read_tail
_safe_state_path = impl._safe_state_path
_task_payload = impl._task_payload

_CHANGES_TELEMETRY_RECENT: dict[tuple[str, str, str, str], float] = {}
_CHANGES_TELEMETRY_TTL_SECONDS = 900.0
_CHANGES_TELEMETRY_MAX_KEYS = 2048
_LOW_CONFIDENCE_FILE_THRESHOLD = 120
_LOW_CONFIDENCE_LINE_THRESHOLD = 4000
_INTERNAL_TASK_METADATA_KEYS = {
    "worktree_dir",
    "task_context",
    "preserved_branch",
    "preserved_base_branch",
    "preserved_base_sha",
    "preserved_head_sha",
    "preserved_merge_base_sha",
    "preserved_at",
    "review_context",
    "execution_checkpoint",
    "pending_precommit_approval",
    "review_stage",
    "active_human_guidance",
    "retry_guidance",
    "requested_changes",
    "pipeline_phase",
    "step_outputs",
}


def _gate_approved_message(gate: str | None, *, will_resume: bool) -> str:
    normalized = str(gate or "").strip()
    if will_resume:
        if normalized == "before_implement":
            return "Plan approved. Task will resume shortly."
        if normalized == "before_generate_tasks":
            return "Task generation approved. Task will resume shortly."
        if normalized == "before_done":
            return "Marked done. Task will finalize shortly."
        if normalized == "after_implement":
            return "Implementation approved. Task will resume shortly."
        if normalized == "before_commit":
            return "Approved. Task will resume shortly."
        if normalized == "human_intervention":
            return "Intervention acknowledged. Task will resume shortly."
        gate_label = _gate_display_label(normalized)
        if gate_label:
            return f"{gate_label} approved. Task will resume shortly."
        return "Gate approved. Task will resume shortly."

    if normalized == "human_intervention":
        return "Intervention acknowledged. Task is still blocked; retry when ready."
    gate_label = _gate_display_label(normalized)
    if gate_label:
        return f"{gate_label} approved."
    return "Gate approved."


def _gate_changes_requested_message(gate: str | None, *, start_from_step: str | None) -> str:
    normalized = str(gate or "").strip()
    if normalized == "before_implement":
        return "Changes requested. Task resumed from planning."
    if normalized == "before_generate_tasks":
        return "Changes requested. Task resumed before task generation."
    if normalized == "before_done":
        return "Changes requested. Task resumed from the final step."
    if start_from_step:
        return f"Changes requested. Task resumed from {start_from_step}."
    return "Changes requested. Task resumed."


def _resume_step_for_gate(gate: str | None) -> str | None:
    mapping = {
        "before_plan": "plan",
        "before_implement": "implement",
        "before_generate_tasks": "generate_tasks",
        "before_done": "__before_done__",
        "after_implement": "review",
        "before_commit": "commit",
    }
    return mapping.get(str(gate or "").strip())


def _pipeline_steps(task: Task) -> list[str]:
    if task.pipeline_template:
        return [str(step).strip() for step in task.pipeline_template if str(step).strip()]
    try:
        return PipelineRegistry().resolve_for_task_type(task.task_type).step_names()
    except Exception:
        return []


def _task_context_manifest_entry(task: Task) -> dict[str, Any] | None:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    context_raw = metadata.get("task_context")
    context = context_raw if isinstance(context_raw, dict) else {}
    retained_worktree = str(context.get("worktree_dir") or metadata.get("worktree_dir") or "").strip()
    task_branch = str(context.get("task_branch") or "").strip()
    preserved_branch = str(metadata.get("preserved_branch") or "").strip()
    has_context = bool(retained_worktree or task_branch or preserved_branch)
    if not has_context:
        return None
    return {
        "task_id": task.id,
        "status": task.status,
        "run_ids": list(task.run_ids or []),
        "retained_worktree_dir": retained_worktree or None,
        "task_branch": task_branch or None,
        "preserved_branch": preserved_branch or None,
        "preserved_base_branch": str(metadata.get("preserved_base_branch") or "").strip() or None,
        "preserved_base_sha": str(metadata.get("preserved_base_sha") or "").strip() or None,
        "preserved_head_sha": str(metadata.get("preserved_head_sha") or "").strip() or None,
        "preserved_merge_base_sha": str(metadata.get("preserved_merge_base_sha") or "").strip() or None,
        "retained_reason": str(context.get("retained_reason") or "").strip() or None,
        "retained_at": str(context.get("retained_at") or "").strip() or None,
        "expected_on_retry": bool(context.get("expected_on_retry")),
        "archived_at": now_iso(),
    }


def _collect_task_context_manifest_entries(tasks: list[Task]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for task in tasks:
        entry = _task_context_manifest_entry(task)
        if entry is not None:
            entries.append(entry)
    return entries


def _sanitize_client_task_metadata(raw: dict[str, Any] | None) -> dict[str, Any]:
    metadata = dict(raw or {})
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized = str(key)
        if normalized in _INTERNAL_TASK_METADATA_KEYS or normalized.startswith("preserved_"):
            continue
        sanitized[normalized] = value
    return sanitized


def _previous_step(steps: list[str], target_step: str) -> str | None:
    if target_step not in steps:
        return None
    index = steps.index(target_step)
    if index <= 0:
        return target_step
    return steps[index - 1]


def _request_changes_step_for_gate(task: Task, gate: str | None) -> str | None:
    normalized = str(gate or "").strip()
    steps = _pipeline_steps(task)
    if normalized == "before_implement":
        return _previous_step(steps, "implement")
    if normalized == "before_generate_tasks":
        return _previous_step(steps, "generate_tasks")
    if normalized == "before_done":
        return steps[-1] if steps else task.current_step
    if normalized == "before_commit":
        return _previous_step(steps, "commit") or "implement"
    if normalized == "after_implement":
        return "implement"
    if normalized == "before_plan":
        return _previous_step(steps, "plan") or "plan"
    return task.current_step or None


def _guidance_fallback_step(task: Task, *, target_step: str | None) -> str | None:
    if str(target_step or "").strip() == "implement_fix":
        return None
    pipeline_steps = set(_pipeline_steps(task))
    if any(step in pipeline_steps for step in {"implement", "prototype", "verify", "benchmark", "reproduce", "review", "commit"}):
        return "implement_fix"
    return None


def _effective_retry_target_step(task: Task) -> str | None:
    if isinstance(task.metadata, dict):
        retry_from = str(task.metadata.get("retry_from_step") or "").strip()
        if retry_from:
            return retry_from
    pipeline_steps = _pipeline_steps(task)
    return pipeline_steps[0] if pipeline_steps else None


def _mark_latest_run_interrupted(container: Any, task: Task, *, summary: str, ts: str) -> None:
    latest_run = None
    for run_id in reversed(task.run_ids):
        latest_run = container.runs.get(run_id)
        if latest_run:
            break
    if latest_run and latest_run.status in {"in_review", "in_progress", "waiting_gate"}:
        latest_run.status = "interrupted"
        latest_run.finished_at = latest_run.finished_at or ts
        latest_run.summary = latest_run.summary or summary
        container.runs.upsert(latest_run)


def _parse_stat_files(stat_text: str) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for line in str(stat_text or "").strip().splitlines():
        text = line.strip()
        if not text or text.startswith("Showing") or " | " not in text:
            continue
        parts = text.split(" | ", 1)
        files.append({"path": parts[0].strip(), "changes": parts[1].strip() if len(parts) > 1 else ""})
    return files


def _is_within_project(candidate: Path, project_root: Path) -> bool:
    try:
        resolved_candidate = candidate.resolve()
        resolved_root = project_root.resolve()
    except Exception:
        return False
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _is_valid_task_worktree_path(
    *,
    task_id: str,
    candidate: Path,
    project_root: Path,
    state_root: Path,
    strict_retained: bool,
) -> bool:
    try:
        resolved_candidate = candidate.resolve()
        resolved_root = project_root.resolve()
        resolved_state = state_root.resolve()
    except Exception:
        return False
    if strict_retained:
        expected = (resolved_state / "worktrees" / str(task_id)).resolve()
        if resolved_candidate != expected:
            return False
    else:
        if not (resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents):
            return False
    if not resolved_candidate.exists() or not resolved_candidate.is_dir():
        return False
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=resolved_candidate,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return False
    return inside.returncode == 0 and str(inside.stdout or "").strip().lower() == "true"


def _resolve_worktree_diff_base_ref(
    *,
    task: Task,
    metadata: dict[str, Any],
    git_dir: Path,
    head_ref: str | None,
) -> tuple[str | None, str]:
    task_context_raw = metadata.get("task_context")
    task_context = task_context_raw if isinstance(task_context_raw, dict) else {}
    preferred_base = str(task_context.get("base_branch") or metadata.get("preserved_base_branch") or "").strip()
    if preferred_base and _resolve_commit_sha(git_dir, preferred_base):
        return preferred_base, "task_context"
    review_context_raw = metadata.get("review_context")
    review_context = review_context_raw if isinstance(review_context_raw, dict) else {}
    review_base = str(review_context.get("base_branch") or "").strip()
    if review_base and _resolve_commit_sha(git_dir, review_base):
        return review_base, "review_context"
    heuristic = _resolve_base_branch(git_dir, avoid_branch=str(head_ref or "").strip() or None)
    if heuristic and _resolve_commit_sha(git_dir, heuristic):
        return heuristic, "heuristic"
    return None, "none"


def _is_runtime_state_path(path_text: str) -> bool:
    normalized = str(path_text or "").replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized == ".agent_orchestrator" or normalized.startswith(".agent_orchestrator/")


def _short_sha(value: str | None) -> str:
    raw = str(value or "").strip()
    return raw[:12] if raw else ""


def _resolve_commit_sha(git_dir: Path, ref: str | None) -> str | None:
    normalized = str(ref or "").strip()
    if not normalized:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{normalized}^{{commit}}"],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    if result.returncode != 0 or not sha:
        return None
    return sha


def _merge_base_sha(git_dir: Path, left_sha: str | None, right_sha: str | None) -> str | None:
    left = str(left_sha or "").strip()
    right = str(right_sha or "").strip()
    if not left or not right:
        return None
    try:
        result = subprocess.run(
            ["git", "merge-base", left, right],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    merge_sha = result.stdout.strip()
    if result.returncode != 0 or not merge_sha:
        return None
    return merge_sha


def _parse_shortstat_counts(text: str) -> tuple[int, int, int]:
    raw = str(text or "").strip()
    if not raw:
        return 0, 0, 0
    files = 0
    additions = 0
    deletions = 0
    file_match = re.search(r"(\d+)\s+files?\s+changed", raw)
    if file_match:
        files = int(file_match.group(1))
    add_match = re.search(r"(\d+)\s+insertions?\(\+\)", raw)
    if add_match:
        additions = int(add_match.group(1))
    del_match = re.search(r"(\d+)\s+deletions?\(-\)", raw)
    if del_match:
        deletions = int(del_match.group(1))
    return files, additions, deletions


def _emit_changes_telemetry_once(
    bus: Any,
    *,
    task_id: str,
    event_type: str,
    base_sha: str | None,
    head_sha: str | None,
    payload: dict[str, Any],
) -> None:
    if bus is None:
        return
    key = (
        str(event_type).strip(),
        str(task_id).strip(),
        str(base_sha or "").strip(),
        str(head_sha or "").strip(),
    )
    now_monotonic = time.monotonic()

    if _CHANGES_TELEMETRY_RECENT:
        stale_keys = [
            existing
            for existing, emitted_at in _CHANGES_TELEMETRY_RECENT.items()
            if now_monotonic - emitted_at > _CHANGES_TELEMETRY_TTL_SECONDS
        ]
        for stale in stale_keys:
            _CHANGES_TELEMETRY_RECENT.pop(stale, None)

    existing_ts = _CHANGES_TELEMETRY_RECENT.get(key)
    if existing_ts is not None and (now_monotonic - existing_ts) <= _CHANGES_TELEMETRY_TTL_SECONDS:
        return

    _CHANGES_TELEMETRY_RECENT[key] = now_monotonic
    if len(_CHANGES_TELEMETRY_RECENT) > _CHANGES_TELEMETRY_MAX_KEYS:
        oldest = sorted(_CHANGES_TELEMETRY_RECENT.items(), key=lambda item: item[1])[: max(1, _CHANGES_TELEMETRY_MAX_KEYS // 8)]
        for old_key, _ in oldest:
            _CHANGES_TELEMETRY_RECENT.pop(old_key, None)

    try:
        bus.emit(
            channel="tasks",
            event_type=event_type,
            entity_id=task_id,
            payload=payload,
        )
    except Exception:
        # Telemetry must never fail the changes endpoint.
        return


def _resolve_base_branch(git_dir: Path, *, avoid_branch: str | None = None) -> str | None:
    candidates: list[str] = []
    try:
        remote_head = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if remote_head.returncode == 0:
            value = remote_head.stdout.strip()
            if value.startswith("origin/"):
                value = value[len("origin/") :]
            if value:
                candidates.append(value)
    except Exception:
        pass

    candidates.extend(["main", "master", "develop", "trunk"])
    try:
        configured_default = subprocess.run(
            ["git", "config", "--get", "init.defaultBranch"],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip()
        if configured_default:
            candidates.append(configured_default)
    except Exception:
        pass
    try:
        current = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        current_name = current.stdout.strip()
        if (
            current.returncode == 0
            and current_name
            and current_name != "HEAD"
            and current_name != str(avoid_branch or "").strip()
        ):
            candidates.append(current_name)
    except Exception:
        pass
    try:
        local_heads = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if local_heads.returncode == 0:
            for raw in local_heads.stdout.splitlines():
                name = raw.strip()
                if not name or name == str(avoid_branch or "").strip():
                    continue
                if name.startswith("task-"):
                    continue
                candidates.append(name)
    except Exception:
        pass

    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen or name == str(avoid_branch or "").strip():
            continue
        seen.add(name)
        try:
            verify = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/heads/{name}"],
                cwd=git_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if verify.returncode == 0:
                return name
        except Exception:
            continue
    return None


def _preserved_branch_changes(
    *,
    task_id: str,
    metadata: dict[str, Any] | None,
    git_dir: Path,
    bus: Any,
) -> dict[str, Any] | None:
    metadata_obj = metadata if isinstance(metadata, dict) else {}
    review_context_raw = metadata_obj.get("review_context")
    review_context: dict[str, Any] = review_context_raw if isinstance(review_context_raw, dict) else {}

    preserved_branch = str((metadata_obj or {}).get("preserved_branch") or "").strip()
    if not preserved_branch:
        return None

    head_sha = _resolve_commit_sha(git_dir, str(metadata_obj.get("preserved_head_sha") or "").strip())
    if not head_sha:
        head_sha = _resolve_commit_sha(git_dir, f"refs/heads/{preserved_branch}")
    if not head_sha:
        return None

    base_ref: str | None = None
    base_sha: str | None = None
    base_source = "none"
    confidence: str = "low"

    metadata_base_sha = _resolve_commit_sha(git_dir, str(metadata_obj.get("preserved_base_sha") or "").strip())
    metadata_base_branch = str(metadata_obj.get("preserved_base_branch") or "").strip()
    review_base_sha = _resolve_commit_sha(git_dir, str(review_context.get("base_sha") or "").strip())
    review_base_branch = str(review_context.get("base_branch") or "").strip()

    if metadata_base_sha:
        base_sha = metadata_base_sha
        base_ref = metadata_base_branch or metadata_base_sha
        base_source = "metadata_sha"
        confidence = "high"

    if not base_sha and metadata_base_branch:
        resolved = _resolve_commit_sha(git_dir, metadata_base_branch)
        if resolved:
            base_sha = resolved
            base_ref = metadata_base_branch
            base_source = "metadata_branch"
            confidence = "high"

    if not base_sha and review_base_sha:
        base_sha = review_base_sha
        base_ref = review_base_branch or review_base_sha
        base_source = "review_context_sha"
        confidence = "medium"

    if not base_sha and review_base_branch:
        resolved = _resolve_commit_sha(git_dir, review_base_branch)
        if resolved:
            base_sha = resolved
            base_ref = review_base_branch
            base_source = "review_context_branch"
            confidence = "medium"

    if not base_sha:
        heuristic_base_branch = _resolve_base_branch(git_dir, avoid_branch=preserved_branch)
        if heuristic_base_branch:
            resolved = _resolve_commit_sha(git_dir, heuristic_base_branch)
            if resolved:
                base_sha = resolved
                base_ref = heuristic_base_branch
                base_source = "heuristic"
                confidence = "low"

    if not base_sha:
        return None

    diff_range = f"{base_sha}..{head_sha}"
    try:
        branch_stat = subprocess.run(
            ["git", "diff", "--stat", diff_range],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        branch_diff = subprocess.run(
            ["git", "diff", diff_range],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        shortstat_result = subprocess.run(
            ["git", "diff", "--shortstat", diff_range],
            cwd=git_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    branch_files = _parse_stat_files(branch_stat.stdout)
    branch_diff_text = branch_diff.stdout or ""
    branch_stat_text = branch_stat.stdout or ""
    if not branch_diff_text and not branch_files and not branch_stat_text.strip():
        return None

    warnings: list[str] = []
    explicit_merge_base_sha = _resolve_commit_sha(git_dir, str(metadata_obj.get("preserved_merge_base_sha") or "").strip())
    computed_merge_base_sha = explicit_merge_base_sha or _merge_base_sha(git_dir, base_sha, head_sha)

    if base_source == "heuristic":
        warnings.append("heuristic_base_inferred")
        files_changed, additions, deletions = _parse_shortstat_counts(shortstat_result.stdout if shortstat_result else "")
        if files_changed <= 0:
            files_changed = len(branch_files)
        if files_changed >= _LOW_CONFIDENCE_FILE_THRESHOLD:
            warnings.append("large_file_count")
        if (additions + deletions) >= _LOW_CONFIDENCE_LINE_THRESHOLD:
            warnings.append("large_line_churn")
        if computed_merge_base_sha and computed_merge_base_sha != base_sha:
            warnings.append("base_not_ancestor")
        confidence = "low"

        _emit_changes_telemetry_once(
            bus,
            task_id=task_id,
            event_type="task.changes_heuristic_base_used",
            base_sha=base_sha,
            head_sha=head_sha,
            payload={
                "base_source": base_source,
                "base_ref": base_ref,
                "base_sha": base_sha,
                "head_ref": preserved_branch,
                "head_sha": head_sha,
                "warnings": list(warnings),
            },
        )

    if warnings:
        _emit_changes_telemetry_once(
            bus,
            task_id=task_id,
            event_type="task.changes_low_confidence",
            base_sha=base_sha,
            head_sha=head_sha,
            payload={
                "base_source": base_source,
                "confidence": confidence,
                "warnings": list(warnings),
                "base_ref": base_ref,
                "base_sha": base_sha,
                "head_ref": preserved_branch,
                "head_sha": head_sha,
            },
        )

    display_base_branch = str(base_ref or "").strip() or None
    return {
        "mode": "preserved_branch",
        "context_source": "preserved_branch",
        "commit": None,
        "branch": preserved_branch,
        "base_branch": display_base_branch,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "head_ref": preserved_branch,
        "head_sha": head_sha,
        "base_source": base_source,
        "confidence": confidence,
        "warnings": warnings,
        "files": branch_files,
        "diff": branch_diff_text,
        "stat": branch_stat_text,
    }


def _timestamp_sort_value(value: Any, *, descending: bool) -> float:
    """Convert ISO timestamps into sortable numeric values."""
    raw = str(value or "").strip()
    if not raw:
        return float("inf")
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return float("inf")
    stamp = parsed.timestamp()
    return -stamp if descending else stamp


def _board_item_sort_key(status: str, item: dict[str, Any]) -> tuple[Any, ...]:
    """Return status-aware sort keys for Kanban board columns."""
    priority = _priority_rank(str(item.get("priority") or "P3"))
    created_at = _timestamp_sort_value(item.get("created_at"), descending=False)
    updated_at_asc = _timestamp_sort_value(item.get("updated_at"), descending=False)
    updated_at_desc = _timestamp_sort_value(item.get("updated_at"), descending=True)
    task_id = str(item.get("id") or "")
    if status in {"backlog", "queued"}:
        return (priority, created_at, updated_at_asc, task_id)
    if status == "in_progress":
        return (priority, updated_at_desc, created_at, task_id)
    if status == "in_review":
        return (priority, updated_at_asc, created_at, task_id)
    if status == "blocked":
        return (priority, updated_at_desc, created_at, task_id)
    if status == "done":
        return (updated_at_desc, priority, created_at, task_id)
    if status == "cancelled":
        return (updated_at_desc, priority, created_at, task_id)
    return (priority, created_at, updated_at_asc, task_id)


def _remove_task_relationship_refs(*, task_id: str, container: Any) -> None:
    """Remove references to a deleted task from all remaining tasks."""
    for existing in container.tasks.list():
        changed = False
        if task_id in existing.blocked_by:
            existing.blocked_by = [dep_id for dep_id in existing.blocked_by if dep_id != task_id]
            changed = True
        if task_id in existing.blocks:
            existing.blocks = [dep_id for dep_id in existing.blocks if dep_id != task_id]
            changed = True
        if existing.parent_id == task_id:
            existing.parent_id = None
            changed = True
        if task_id in existing.children_ids:
            existing.children_ids = [child_id for child_id in existing.children_ids if child_id != task_id]
            changed = True
        if changed:
            existing.updated_at = now_iso()
            container.tasks.upsert(existing)


def _set_task_step_timeout(task: Task, timeout_seconds: int | None) -> None:
    """Set or clear per-task implement timeout metadata."""
    if not isinstance(task.metadata, dict):
        task.metadata = {}
    metadata = task.metadata
    existing = metadata.get("step_timeouts")
    step_timeouts: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    if timeout_seconds is None:
        step_timeouts.pop("implement", None)
        step_timeouts.pop("implement_fix", None)
        if step_timeouts:
            metadata["step_timeouts"] = step_timeouts
        else:
            metadata.pop("step_timeouts", None)
        return
    step_timeouts["implement"] = int(timeout_seconds)
    metadata["step_timeouts"] = step_timeouts


def register_task_routes(router: APIRouter, deps: RouteDeps) -> None:
    """Register task, planning, execution, and logs routes."""
    @router.post("/tasks")
    async def create_task(body: CreateTaskRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Create and persist a task with resolved pipeline configuration.
        
        Args:
            body: Request payload describing task metadata and pipeline preferences.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created task details.
        
        Raises:
            HTTPException: If task type auto-resolution inputs are invalid.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        registry = PipelineRegistry()
        allowed_pipelines = sorted(template.id for template in registry.list_templates())
        classifier_pipeline_id = str(body.classifier_pipeline_id or "").strip()
        classifier_confidence = str(body.classifier_confidence or "").strip().lower()
        classifier_reason = str(body.classifier_reason or "").strip()
        classifier_pipeline_valid = classifier_pipeline_id in allowed_pipelines
        classifier_confidence_valid = classifier_confidence in {"high", "low"}
        requested_task_type = str(body.task_type or "feature").strip() or "feature"

        if requested_task_type == "auto":
            if classifier_confidence != "high" or not classifier_pipeline_valid:
                raise HTTPException(
                    status_code=400,
                    detail="Auto task type requires a high-confidence classifier result.",
                )
            resolved_task_type = _canonical_task_type_for_pipeline(registry, classifier_pipeline_id)
        else:
            resolved_task_type = requested_task_type

        template = registry.resolve_for_task_type(resolved_task_type)
        final_pipeline_id = template.id
        pipeline_steps = body.pipeline_template
        if pipeline_steps is None:
            pipeline_steps = template.step_names()
        dep_policy = body.dependency_policy.strip() if body.dependency_policy else ""
        if dep_policy not in ("permissive", "prudent", "strict"):
            cfg = container.config.load()
            dep_policy = str((cfg.get("defaults") or {}).get("dependency_policy") or "prudent")
            if dep_policy not in ("permissive", "prudent", "strict"):
                dep_policy = "prudent"
        cfg = container.config.load()
        configured_hitl_mode = normalize_hitl_mode((cfg.get("defaults") or {}).get("hitl_mode"))
        requested_hitl_mode = normalize_hitl_mode(body.hitl_mode, default="")
        hitl_mode = requested_hitl_mode or configured_hitl_mode
        priority = body.priority if body.priority in ("P0", "P1", "P2", "P3") else "P2"
        task = Task(
            title=body.title,
            description=body.description,
            task_type=resolved_task_type,
            priority=cast(Priority, priority),
            labels=body.labels,
            blocked_by=body.blocked_by,
            parent_id=body.parent_id,
            pipeline_template=pipeline_steps,
            hitl_mode=hitl_mode,
            dependency_policy=cast(DependencyPolicy, dep_policy),
            source=body.source,
            worker_model=(str(body.worker_model).strip() if body.worker_model else None),
            metadata=_sanitize_client_task_metadata(body.metadata),
            project_commands=(body.project_commands if body.project_commands else None),
        )
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        if body.step_timeout_seconds is not None:
            _set_task_step_timeout(task, body.step_timeout_seconds)
        if classifier_pipeline_valid and classifier_confidence_valid:
            task.metadata["classifier_pipeline_id"] = classifier_pipeline_id
            task.metadata["classifier_confidence"] = classifier_confidence
            task.metadata["classifier_reason"] = classifier_reason
            task.metadata["was_user_override"] = bool(body.was_user_override)
        task.metadata["final_pipeline_id"] = final_pipeline_id
        requested_status = str(body.status or "").strip()
        valid_statuses = {"backlog", "queued", "in_progress", "in_review", "done", "blocked", "cancelled"}
        if requested_status in valid_statuses:
            task.status = cast(TaskStatus, requested_status)
        elif not requested_status:
            # Keep API create deterministic by default; callers can opt into
            # immediate scheduler pickup by explicitly passing status=queued.
            task.status = "backlog"
        if task.parent_id:
            parent = container.tasks.get(task.parent_id)
            if parent and task.id not in parent.children_ids:
                parent.children_ids.append(task.id)
                container.tasks.upsert(parent)
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.created", entity_id=task.id, payload={"status": task.status})
        return {"task": _task_payload(task, orchestrator=orchestrator)}

    @router.post("/tasks/classify-pipeline", response_model=PipelineClassificationResponse)
    async def classify_pipeline(
        body: PipelineClassificationRequest,
        project_dir: Optional[str] = Query(None),
    ) -> PipelineClassificationResponse:
        """Classify the best pipeline template for a proposed task.
        
        Args:
            body: Request payload with title, description, and optional metadata.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A normalized pipeline classification result with confidence and rationale.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        registry = PipelineRegistry()
        allowed_pipelines = sorted(template.id for template in registry.list_templates())
        synthetic = Task(
            title=str(body.title or "").strip(),
            description=str(body.description or "").strip(),
            task_type="feature",
            status="queued",
            metadata=_sanitize_client_task_metadata(body.metadata),
        )
        if not isinstance(synthetic.metadata, dict):
            synthetic.metadata = {}
        synthetic.metadata["classification_allowed_pipelines"] = allowed_pipelines
        try:
            result = orchestrator.worker_adapter.run_step_ephemeral(task=synthetic, step="pipeline_classify", attempt=1)
        except Exception:
            result = None
        if result is None or result.status != "ok":
            return PipelineClassificationResponse(
                pipeline_id="feature",
                task_type="feature",
                confidence="low",
                reason="Pipeline auto-classification failed. Please select a pipeline.",
                allowed_pipelines=allowed_pipelines,
            )
        return _normalize_pipeline_classification_output(
            summary=result.summary,
            allowed_pipelines=allowed_pipelines,
            registry=registry,
        )

    @router.get("/tasks")
    async def list_tasks(
        project_dir: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        task_type: Optional[str] = Query(None),
        priority: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """List tasks filtered by lifecycle, type, or priority.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
            status: Optional status filter.
            task_type: Optional task type filter.
            priority: Optional priority filter.
        
        Returns:
            A payload with sorted task summaries and total count.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        tasks = container.tasks.list()
        filtered = []
        for task in tasks:
            if status and task.status != status:
                continue
            if task_type and task.task_type != task_type:
                continue
            if priority and task.priority != priority:
                continue
            filtered.append(task)
        filtered.sort(key=lambda t: (_priority_rank(t.priority), t.created_at))
        return {
            "tasks": [_task_payload(task, orchestrator=orchestrator) for task in filtered],
            "total": len(filtered),
        }

    @router.get("/tasks/board")
    async def board(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return task data grouped by Kanban-style status columns.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload keyed by task status with sorted task cards.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        columns: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ["backlog", "queued", "in_progress", "in_review", "blocked", "done", "cancelled"]
        }
        for task in container.tasks.list():
            columns.setdefault(task.status, []).append(_task_payload(task, orchestrator=orchestrator))
        for status, items in columns.items():
            items.sort(key=lambda item: _board_item_sort_key(status, item))
        return {"columns": columns}

    @router.post("/tasks/clear")
    async def clear_tasks(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Clear all tasks by archiving runtime state and reinitializing storage.

        Args:
            project_dir: Optional project directory used to resolve runtime state.

        Returns:
            A payload indicating clear status and archive destination path.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        # Pause intake and stop scheduler/workers before mutating state files.
        orchestrator.control("pause")
        orchestrator.shutdown(timeout=10.0)

        context_entries = _collect_task_context_manifest_entries(container.tasks.list())
        context_manifest = archive_task_context_manifest(container.project_dir, context_entries)
        archived_to = archive_state_root(container.project_dir)
        ensure_state_root(container.project_dir)
        deps.job_store.clear()
        # Clearing state recreates config defaults; ensure queue intake returns
        # to running mode so queued tasks cannot get stranded in paused mode.
        orchestrator.control("resume")
        archive_path = str(archived_to) if archived_to else ""
        message = (
            f"Cleared all tasks. Archived previous runtime state to {archive_path}."
            if archive_path
            else "Cleared all tasks. No existing runtime state archive was needed."
        )
        payload = {
            "archived_to": archive_path,
            "message": message,
            "cleared_at": now_iso(),
            "archived_context_count": len(context_entries),
            "archived_context_manifest_path": str(context_manifest) if context_manifest else "",
        }
        bus.emit(channel="tasks", event_type="tasks.cleared", entity_id=container.project_id, payload=payload)
        bus.emit(channel="notifications", event_type="tasks.cleared", entity_id=container.project_id, payload=payload)
        return {"cleared": True, **payload}

    @router.get("/tasks/execution-order")
    async def execution_order(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Compute execution batches for non-terminal tasks.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing ready-to-run batches and recently completed tasks.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        tasks = container.tasks.list()
        terminal = {"done", "cancelled"}
        pending = [t for t in tasks if t.status not in terminal]
        completed = [t for t in tasks if t.status in terminal]
        completed.sort(key=lambda t: t.updated_at, reverse=True)
        return {
            "batches": _execution_batches(pending),
            "completed": [t.id for t in completed],
        }

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Load one task and return its expanded payload.
        
        Args:
            task_id: Identifier of the task to fetch.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the requested task.
        
        Raises:
            HTTPException: If the task does not exist.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": _task_payload(task, container, orchestrator)}

    @router.delete("/tasks/{task_id}")
    async def delete_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Delete a terminal task and clean stale task relationship references.

        Args:
            task_id: Identifier of the task to delete.
            project_dir: Optional project directory used to resolve runtime state.

        Returns:
            A payload indicating successful deletion.

        Raises:
            HTTPException: If the task is missing or non-terminal.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.status not in {"done", "cancelled"}:
            raise HTTPException(status_code=400, detail="Only terminal tasks (done/cancelled) can be deleted.")

        context_entry = _task_context_manifest_entry(task)
        context_manifest = (
            archive_task_context_manifest(container.project_dir, [context_entry])
            if context_entry is not None
            else None
        )
        _remove_task_relationship_refs(task_id=task_id, container=container)
        if not container.tasks.delete(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        bus.emit(channel="tasks", event_type="task.deleted", entity_id=task_id, payload={"status": task.status})
        return {
            "deleted": True,
            "task_id": task_id,
            "archived_context_count": 1 if context_entry is not None else 0,
            "archived_context_manifest_path": str(context_manifest) if context_manifest else "",
        }

    @router.get("/tasks/{task_id}/diff")
    async def get_task_diff(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Load git diff metadata for the most recent commit step on a task.

        Args:
            task_id: Identifier of the task whose commit diff should be loaded.
            project_dir: Optional project directory used to resolve runtime
                state and git context.

        Returns:
            A payload with ``commit``, ``files``, ``diff``, and ``stat`` keys.
            When no commit step exists, returns
            ``{"commit": None, "files": [], "diff": "", "stat": ""}``.

        Raises:
            HTTPException: If the task does not exist (404) or git diff/stat
                retrieval fails (500).
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Find the commit SHA from the latest run's steps.
        commit_sha: str | None = None
        for run_id in reversed(task.run_ids):
            for run in container.runs.list():
                if run.id == run_id:
                    for step in reversed(run.steps or []):
                        if isinstance(step, dict) and step.get("step") == "commit" and step.get("commit"):
                            commit_sha = str(step["commit"])
                            break
                    if commit_sha:
                        break
            if commit_sha:
                break

        if not commit_sha:
            return {"commit": None, "files": [], "diff": "", "stat": ""}

        git_dir = container.project_dir
        try:
            stat_result = subprocess.run(
                ["git", "show", "--stat", "--format=", commit_sha],
                cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
            )
            diff_result = subprocess.run(
                ["git", "diff", f"{commit_sha}~1..{commit_sha}"],
                cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read diff: {exc}") from exc

        files = _parse_stat_files(stat_result.stdout)

        return {
            "commit": commit_sha,
            "files": files,
            "diff": diff_result.stdout,
            "stat": stat_result.stdout,
        }

    @router.get("/tasks/{task_id}/changes")
    async def get_task_changes(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return best-available task change evidence.

        Prefers committed diff when present; otherwise returns current working tree
        changes for task worktree/repo context.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        task_metadata = task.metadata if isinstance(task.metadata, dict) else {}
        worktree_dir = None
        worktree_candidate_present = False
        if isinstance(task_metadata, dict):
            candidate = str(task_metadata.get("worktree_dir") or "").strip()
            if candidate:
                worktree_candidate_present = True
                worktree_dir = candidate
        git_dir: Path | None = None
        project_root = container.project_dir
        has_task_worktree = False
        strict_retained_context = task.status == "blocked"
        if worktree_dir:
            try:
                worktree_path = Path(worktree_dir).expanduser().resolve()
            except Exception:
                worktree_path = None
            if (
                worktree_path
                and _is_valid_task_worktree_path(
                    task_id=task.id,
                    candidate=worktree_path,
                    project_root=project_root,
                    state_root=container.state_root,
                    strict_retained=strict_retained_context,
                )
            ):
                git_dir = worktree_path
                has_task_worktree = True
        prefer_retained_worktree = bool(task.status == "blocked" and has_task_worktree)

        if not prefer_retained_worktree:
            commit_payload: dict[str, Any] | None = None
            try:
                commit_payload = await get_task_diff(task_id, project_dir)
            except HTTPException as exc:
                if exc.status_code != 500:
                    raise
            if commit_payload and commit_payload.get("commit"):
                return {
                    "mode": "committed",
                    "context_source": "committed",
                    "commit": commit_payload.get("commit"),
                    "base_ref": None,
                    "base_sha": None,
                    "head_ref": None,
                    "head_sha": None,
                    "base_source": "none",
                    "confidence": "high",
                    "warnings": [],
                    "files": commit_payload.get("files") or [],
                    "diff": commit_payload.get("diff") or "",
                    "stat": commit_payload.get("stat") or "",
                }

        if has_task_worktree and git_dir is not None:
            try:
                status_result = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
                )
                head_ref_result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=git_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                head_ref: str | None = str(head_ref_result.stdout or "").strip()
                if head_ref_result.returncode != 0 or not head_ref or head_ref == "HEAD":
                    head_ref = None
                has_head = (
                    subprocess.run(
                        ["git", "rev-parse", "--verify", "HEAD"],
                        cwd=git_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    ).returncode
                    == 0
                )
                base_ref = None
                base_source = "none"
                if prefer_retained_worktree and has_head:
                    base_ref, base_source = _resolve_worktree_diff_base_ref(
                        task=task,
                        metadata=task_metadata,
                        git_dir=git_dir,
                        head_ref=head_ref,
                    )
                if base_ref and has_head:
                    stat_cmd = ["git", "diff", "--stat", base_ref]
                    diff_cmd = ["git", "diff", base_ref]
                else:
                    stat_cmd = ["git", "diff", "--stat", "HEAD"] if has_head else ["git", "diff", "--stat"]
                    diff_cmd = ["git", "diff", "HEAD"] if has_head else ["git", "diff"]
                stat_result = subprocess.run(
                    stat_cmd,
                    cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
                )
                diff_result = subprocess.run(
                    diff_cmd,
                    cwd=git_dir, capture_output=True, text=True, check=True, timeout=10,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                raise HTTPException(status_code=500, detail=f"Failed to read working-tree changes: {exc}") from exc

            raw_status_lines = [line.rstrip() for line in status_result.stdout.splitlines() if line.strip()]
            status_entries: list[dict[str, str]] = []
            for line in raw_status_lines:
                code = line[:2].strip() or "??"
                path_fragment = line[3:].strip() if len(line) > 3 else line.strip()
                logical_paths = [path_fragment]
                if " -> " in path_fragment:
                    logical_paths = [p.strip() for p in path_fragment.split(" -> ", 1)]
                if logical_paths and all(_is_runtime_state_path(p) for p in logical_paths):
                    continue
                status_entries.append({"code": code, "path": path_fragment})
            status_lines = [f"{entry['code']} {entry['path']}".strip() for entry in status_entries]

            files = [entry for entry in _parse_stat_files(stat_result.stdout) if not _is_runtime_state_path(entry.get("path", ""))]
            if not files and status_entries:
                parsed: list[dict[str, str]] = []
                for entry in status_entries:
                    parsed.append({"path": entry["path"], "changes": entry["code"]})
                files = parsed

            diff_text = diff_result.stdout or ""
            if files or status_lines:
                stat_text = stat_result.stdout or ("\n".join(status_lines) if status_lines else "")
                return {
                    "mode": "working_tree",
                    "context_source": "retained_worktree",
                    "commit": None,
                    "base_ref": base_ref,
                    "base_sha": _resolve_commit_sha(git_dir, base_ref) if base_ref else None,
                    "head_ref": head_ref,
                    "head_sha": _resolve_commit_sha(git_dir, head_ref) if head_ref else None,
                    "base_source": base_source,
                    "confidence": "high",
                    "warnings": [],
                    "files": files,
                    "diff": diff_text,
                    "stat": stat_text,
                }

        if prefer_retained_worktree:
            commit_payload = None
            try:
                commit_payload = await get_task_diff(task_id, project_dir)
            except HTTPException as exc:
                if exc.status_code != 500:
                    raise
            if commit_payload and commit_payload.get("commit"):
                return {
                    "mode": "committed",
                    "context_source": "committed",
                    "commit": commit_payload.get("commit"),
                    "base_ref": None,
                    "base_sha": None,
                    "head_ref": None,
                    "head_sha": None,
                    "base_source": "none",
                    "confidence": "high",
                    "warnings": [],
                    "files": commit_payload.get("files") or [],
                    "diff": commit_payload.get("diff") or "",
                    "stat": commit_payload.get("stat") or "",
                }

        preserved_payload = _preserved_branch_changes(
            task_id=task.id,
            metadata=task_metadata,
            git_dir=container.project_dir,
            bus=bus,
        )
        if preserved_payload:
            return preserved_payload

        if has_task_worktree:
            return {
                "mode": "none",
                "context_source": "none",
                "commit": None,
                "base_ref": None,
                "base_sha": None,
                "head_ref": None,
                "head_sha": None,
                "base_source": "none",
                "confidence": "high",
                "warnings": [],
                "files": [],
                "diff": "",
                "stat": "",
            }

        reason = "task_context_missing"
        if worktree_candidate_present and not has_task_worktree:
            reason = "invalid_worktree_context"
        elif str(task_metadata.get("preserved_branch") or "").strip():
            reason = "preserved_branch_missing"
        return {
            "mode": "none",
            "reason": reason,
            "context_source": "none",
            "commit": None,
            "base_ref": None,
            "base_sha": None,
            "head_ref": None,
            "head_sha": None,
            "base_source": "none",
            "confidence": "low",
            "warnings": [],
            "files": [],
            "diff": "",
            "stat": "",
        }

    @router.get("/tasks/{task_id}/logs")
    async def get_task_logs(
        task_id: str,
        project_dir: Optional[str] = Query(None),
        max_chars: int = Query(12000, ge=200, le=2000000),
        stdout_offset: int = Query(0, ge=0),
        stderr_offset: int = Query(0, ge=0),
        backfill: bool = Query(False),
        stdout_read_to: int = Query(0, ge=0),
        stderr_read_to: int = Query(0, ge=0),
        step: Optional[str] = Query(None),
        run_id: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Return active, historical, or incremental task log output.
        
        Args:
            task_id: Identifier of the task whose logs are requested.
            project_dir: Optional project directory used to resolve runtime state.
            max_chars: Maximum number of characters returned from each stream read.
            stdout_offset: Byte offset for incremental stdout reads.
            stderr_offset: Byte offset for incremental stderr reads.
            backfill: Whether to align incremental reads to full-line boundaries.
            stdout_read_to: Optional upper byte boundary for stdout reads.
            stderr_read_to: Optional upper byte boundary for stderr reads.
            step: Optional step name to load logs from historical run metadata.
            run_id: Optional run identifier used with ``step`` to select a specific
                historical step attempt.
        
        Returns:
            A payload containing log text, offsets, and progress metadata.
        
        Raises:
            HTTPException: If the task does not exist.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        metadata = task.metadata if isinstance(task.metadata, dict) else {}

        active_meta = metadata.get("active_logs") if isinstance(metadata.get("active_logs"), dict) else None
        last_meta = metadata.get("last_logs") if isinstance(metadata.get("last_logs"), dict) else None

        requested_step = str(step or "").strip()
        requested_run_id = str(run_id or "").strip()

        all_runs = container.runs.list()
        runs_by_id = {str(run.id): run for run in all_runs if str(run.id).strip()}
        task_run_ids = [str(run_id_value).strip() for run_id_value in list(task.run_ids or []) if str(run_id_value).strip()]
        if task_run_ids:
            known_run_ids = set(task_run_ids)
            for run in all_runs:
                run_id_value = str(run.id).strip()
                if not run_id_value:
                    continue
                if str(run.task_id or "").strip() != task.id:
                    continue
                if run_id_value in known_run_ids:
                    continue
                task_run_ids.append(run_id_value)
                known_run_ids.add(run_id_value)
        else:
            task_run_ids = [
                str(run.id).strip()
                for run in all_runs
                if str(run.id).strip() and str(run.task_id or "").strip() == task.id
            ]
        # Normalize run ordering using run timestamps so latest-log selection does
        # not depend on historical run_ids list direction.
        indexed_run_ids = {run_id_value: idx for idx, run_id_value in enumerate(task_run_ids)}

        def _run_sort_key(run_id_value: str) -> tuple[float, int, str]:
            run = runs_by_id.get(run_id_value)
            fallback_idx = indexed_run_ids.get(run_id_value, 10_000_000)
            if not run:
                return float("-inf"), fallback_idx, run_id_value
            started_at_raw = str(getattr(run, "started_at", "") or "").strip()
            finished_at_raw = str(getattr(run, "finished_at", "") or "").strip()
            candidate = started_at_raw or finished_at_raw
            if not candidate:
                return float("-inf"), fallback_idx, run_id_value
            normalized = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
            try:
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                parsed_epoch = parsed.astimezone(timezone.utc).timestamp()
            except ValueError:
                parsed_epoch = float("-inf")
            return parsed_epoch, fallback_idx, run_id_value

        task_run_ids = sorted(task_run_ids, key=_run_sort_key)
        step_execution_counts: dict[str, int] = {}
        step_latest_run: dict[str, str] = {}
        available_steps: list[str] = []
        seen_steps: set[str] = set()
        # Most-recent-first list of step executions with available stdout logs.
        step_history_entries: list[dict[str, Any]] = []
        step_attempt_counts: dict[str, int] = {}
        for task_run_id in reversed(task_run_ids):
            selected_run = runs_by_id.get(task_run_id)
            if not selected_run:
                continue
            for entry in reversed(selected_run.steps or []):
                if not isinstance(entry, dict) or not entry.get("stdout_path"):
                    continue
                step_name = str(entry.get("step") or "").strip()
                if not step_name:
                    continue
                step_execution_counts[step_name] = step_execution_counts.get(step_name, 0) + 1
                if step_name not in seen_steps:
                    available_steps.append(step_name)
                    seen_steps.add(step_name)
                if step_name not in step_latest_run:
                    step_latest_run[step_name] = task_run_id
                step_attempt_counts[step_name] = step_attempt_counts.get(step_name, 0) + 1
                step_history_entries.append(
                    {
                        "step": step_name,
                        "run_id": task_run_id,
                        # Temporary recent-first ordinal; normalized below so
                        # attempt numbers are chronological (oldest=1, newest=N).
                        "attempt": step_attempt_counts[step_name],
                        "started_at": entry.get("started_at"),
                        "finished_at": entry.get("ts"),
                        "entry": entry,
                    }
                )

        # Normalize per-step attempt numbering so higher attempt means more recent.
        # `step_history_entries` is recent-first, while `step_execution_counts`
        # carries total executions for each step.
        for item in step_history_entries:
            step_name = str(item.get("step") or "").strip()
            if not step_name:
                continue
            recent_first_attempt = int(item.get("attempt") or 0)
            total_for_step = int(step_execution_counts.get(step_name) or 0)
            if recent_first_attempt > 0 and total_for_step > 0:
                item["attempt"] = total_for_step - recent_first_attempt + 1

        # When a specific step is requested, resolve logs strictly for that step.
        step_logs_meta: Optional[dict[str, Any]] = None
        selected_run_id: str | None = None
        mode = "none"
        if requested_step:
            strict_run_selection = bool(requested_run_id)
            if requested_run_id:
                direct_run = runs_by_id.get(requested_run_id)
                if direct_run:
                    for entry in reversed(direct_run.steps or []):
                        if (
                            isinstance(entry, dict)
                            and str(entry.get("step") or "").strip() == requested_step
                            and entry.get("stdout_path")
                        ):
                            step_logs_meta = entry
                            selected_run_id = requested_run_id
                            break
            for item in step_history_entries:
                if step_logs_meta:
                    break
                if item.get("step") != requested_step:
                    continue
                if requested_run_id and str(item.get("run_id") or "") != requested_run_id:
                    continue
                selected_run_id = str(item.get("run_id") or "") or None
                history_entry = item.get("entry")
                step_logs_meta = history_entry if isinstance(history_entry, dict) else None
                if step_logs_meta:
                    break

            if step_logs_meta:
                logs_meta = step_logs_meta
                mode = "history"
            elif strict_run_selection:
                logs_meta = {"step": requested_step}
            elif active_meta and str(active_meta.get("step") or "").strip() == requested_step:
                logs_meta = active_meta
                selected_run_id = str(active_meta.get("run_id") or "").strip() or None
                mode = "active"
            elif last_meta and str(last_meta.get("step") or "").strip() == requested_step:
                logs_meta = last_meta
                selected_run_id = str(last_meta.get("run_id") or "").strip() or None
                mode = "last"
            else:
                logs_meta = {"step": requested_step}
        else:
            # Keep active stream selection first for in-progress tasks, but for
            # non-scoped requests fall back to the latest historical execution
            # before considering stale `last_logs` metadata.
            latest_history = step_history_entries[0] if step_history_entries else None
            latest_history_entry = latest_history.get("entry") if isinstance(latest_history, dict) else None
            if active_meta:
                logs_meta = active_meta
                selected_run_id = str(logs_meta.get("run_id") or "").strip() or None
                mode = "active"
            elif isinstance(latest_history_entry, dict):
                logs_meta = latest_history_entry
                selected_run_id = str(latest_history.get("run_id") or "").strip() or None
                mode = "history"
            elif last_meta:
                logs_meta = last_meta
                selected_run_id = str(logs_meta.get("run_id") or "").strip() or None
                mode = "last"
            else:
                logs_meta = {}
                mode = "none"

        stdout_path = _safe_state_path(logs_meta.get("stdout_path"), container.state_root)
        stderr_path = _safe_state_path(logs_meta.get("stderr_path"), container.state_root)
        progress_path = _safe_state_path(logs_meta.get("progress_path"), container.state_root)

        progress_payload: dict[str, Any] = {}
        if progress_path and progress_path.exists():
            try:
                raw = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    progress_payload = raw
            except Exception:
                progress_payload = {}

        use_incremental = backfill or stdout_offset > 0 or stderr_offset > 0
        stdout_tail_start = 0
        stderr_tail_start = 0
        stdout_chunk_start = 0
        stderr_chunk_start = 0
        if use_incremental:
            stdout_text, new_stdout_offset, stdout_chunk_start = _read_from_offset(
                stdout_path,
                stdout_offset,
                max_chars,
                read_to=stdout_read_to,
                align_start_to_line=backfill,
            )
            stderr_text, new_stderr_offset, stderr_chunk_start = _read_from_offset(
                stderr_path,
                stderr_offset,
                max_chars,
                read_to=stderr_read_to,
                align_start_to_line=backfill,
            )
        else:
            stdout_text, stdout_tail_start = _read_tail(stdout_path, max_chars)
            stderr_text, stderr_tail_start = _read_tail(stderr_path, max_chars)
            new_stdout_offset = stdout_path.stat().st_size if stdout_path and stdout_path.exists() else 0
            new_stderr_offset = stderr_path.stat().st_size if stderr_path and stderr_path.exists() else 0
            stdout_chunk_start = stdout_tail_start
            stderr_chunk_start = stderr_tail_start

        # Include the currently active log step for live switching support.
        active_step = active_meta.get("step") if isinstance(active_meta, dict) else None
        if active_step and active_step not in seen_steps:
            available_steps.append(str(active_step))
        if active_step and str(active_step) not in step_execution_counts:
            step_execution_counts[str(active_step)] = 1

        return {
            "mode": mode,
            "task_status": task.status,
            "step": str(logs_meta.get("step") or task.current_step or ""),
            "run_id": selected_run_id,
            "current_step": str(task.current_step or ""),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_offset": new_stdout_offset,
            "stderr_offset": new_stderr_offset,
            "stdout_chunk_start": stdout_chunk_start,
            "stderr_chunk_start": stderr_chunk_start,
            "stdout_tail_start": stdout_tail_start,
            "stderr_tail_start": stderr_tail_start,
            "started_at": logs_meta.get("started_at"),
            "finished_at": logs_meta.get("finished_at"),
            "log_id": _logs_snapshot_id(logs_meta),
            "progress": progress_payload,
            "available_steps": available_steps,
            "step_execution_counts": step_execution_counts,
            "step_latest_run": step_latest_run,
            "step_history": [
                {
                    "step": str(item.get("step") or ""),
                    "run_id": str(item.get("run_id") or ""),
                    "attempt": int(item.get("attempt") or 0),
                    "started_at": item.get("started_at"),
                    "finished_at": item.get("finished_at"),
                }
                for item in step_history_entries[:200]
            ],
        }

    @router.patch("/tasks/{task_id}")
    async def patch_task(task_id: str, body: UpdateTaskRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Apply mutable field updates to an existing task.
        
        Args:
            task_id: Identifier of the task to update.
            body: Partial task update payload.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated task.
        
        Raises:
            HTTPException: If the task is missing or if status mutation is requested.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if "step_timeout_seconds" in body.model_fields_set:
            _set_task_step_timeout(task, body.step_timeout_seconds)
        updates = body.model_dump(exclude_none=True, exclude={"step_timeout_seconds"})
        if "status" in updates:
            raise HTTPException(
                status_code=400,
                detail="Task status cannot be changed via PATCH. Use /transition, /retry, /cancel, or review actions.",
            )
        if "metadata" in updates:
            incoming_metadata = updates.get("metadata")
            if not isinstance(incoming_metadata, dict):
                raise HTTPException(status_code=400, detail="Task metadata must be an object.")
            invalid_keys = [
                str(key)
                for key in incoming_metadata.keys()
                if str(key) != "user" and not str(key).startswith("user.")
            ]
            if invalid_keys:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Only metadata.user.* keys are mutable via PATCH. "
                        f"Unsupported keys: {', '.join(sorted(invalid_keys))}"
                    ),
                )
            existing_metadata = task.metadata if isinstance(task.metadata, dict) else {}
            merged_metadata = dict(existing_metadata)
            for key in list(merged_metadata.keys()):
                if str(key) == "user" or str(key).startswith("user."):
                    merged_metadata.pop(key, None)
            for key, value in incoming_metadata.items():
                merged_metadata[str(key)] = value
            updates["metadata"] = merged_metadata
        if "project_commands" in updates and not updates["project_commands"]:
            updates["project_commands"] = None
        for key, value in updates.items():
            if key == "hitl_mode":
                value = normalize_hitl_mode(value)
            setattr(task, key, value)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.updated", entity_id=task.id, payload={"status": task.status})
        return {"task": _task_payload(task, container, orchestrator)}

    @router.post("/tasks/{task_id}/transition")
    async def transition_task(task_id: str, body: TransitionRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Transition a task to a valid next status.
        
        Args:
            task_id: Identifier of the task to transition.
            body: Requested target status payload.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the transitioned task.
        
        Raises:
            HTTPException: If the task is missing, the transition is invalid, or blockers remain.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        target = body.status
        if task.status == "blocked" and target == "in_review":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Blocked tasks cannot transition to in_review directly. "
                    "Use /tasks/{task_id}/skip-to-precommit when eligible."
                ),
            )
        if task.status == "in_review" and target == "done":
            raise HTTPException(
                status_code=400,
                detail=(
                    "In-review tasks cannot transition to done directly. "
                    "Use /review/{task_id}/approve."
                ),
            )
        valid = VALID_TRANSITIONS.get(task.status, set())
        if target not in valid:
            raise HTTPException(status_code=400, detail=f"Invalid transition {task.status} -> {target}")
        if target == "cancelled":
            try:
                cancelled = orchestrator.cancel_task(task.id, source="api_transition")
            except ValueError as exc:
                if "Task not found" in str(exc):
                    raise HTTPException(status_code=404, detail=str(exc))
                raise HTTPException(status_code=400, detail=str(exc))
            bus.emit(
                channel="tasks",
                event_type="task.transitioned",
                entity_id=cancelled.id,
                payload={"status": cancelled.status},
            )
            return {"task": _task_payload(cancelled, container, orchestrator)}
        if target == "queued":
            unresolved = _has_unresolved_blockers(container, task)
            if unresolved is not None:
                raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        task.status = cast(TaskStatus, target)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.transitioned", entity_id=task.id, payload={"status": task.status})
        return {"task": _task_payload(task, container, orchestrator)}

    @router.post("/tasks/{task_id}/run")
    async def run_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Trigger orchestrator execution for a queued task.
        
        Args:
            task_id: Identifier of the task to execute.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the task after run dispatch.
        
        Raises:
            HTTPException: If the task cannot be run.
        """
        _, _, orchestrator = deps.ctx(project_dir)
        try:
            task = orchestrator.run_task(task_id)
        except ValueError as exc:
            if "Task not found" in str(exc):
                raise HTTPException(status_code=404, detail=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        return {"task": _task_payload(task, orchestrator=orchestrator)}

    @router.post("/tasks/{task_id}/skip-to-precommit")
    async def skip_to_precommit(
        task_id: str,
        body: Optional[SkipToPrecommitRequest] = None,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Move an eligible blocked task directly to pre-commit review.

        Args:
            task_id: Identifier of the blocked task.
            body: Optional payload with reviewer guidance/audit note.
            project_dir: Optional project directory used to resolve runtime state.

        Returns:
            Updated task payload in pre-commit review state.

        Raises:
            HTTPException: If task lookup fails or task is not eligible.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        allowed, reason_code = orchestrator.can_skip_to_precommit(task)
        if not allowed:
            raise HTTPException(
                status_code=409,
                detail=f"Task {task_id} is not eligible to skip to pre-commit ({reason_code or 'not_allowed'})",
            )
        guidance = str(body.guidance if body else "").strip()
        try:
            updated = orchestrator.skip_task_to_precommit(task_id, guidance=guidance)
        except ValueError as exc:
            message = str(exc)
            if "Task not found" in message:
                raise HTTPException(status_code=404, detail=message) from exc
            raise HTTPException(status_code=409, detail=message) from exc
        return {
            "task": _task_payload(updated, container, orchestrator),
            "message": "Task moved to pre-commit review. Approve to run commit.",
        }

    @router.post("/tasks/{task_id}/retry")
    async def retry_task(task_id: str, body: Optional[RetryTaskRequest] = None, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Retry a task after clearing error and gate state.
        
        Args:
            task_id: Identifier of the task to retry.
            body: Optional retry payload containing guidance and restart step.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the retried task.
        
        Raises:
            HTTPException: If the task is missing or unresolved blockers remain.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        unresolved = _has_unresolved_blockers(container, task)
        if unresolved is not None:
            raise HTTPException(status_code=400, detail=f"Unresolved blocker: {unresolved}")
        # Capture previous error before clearing it.
        previous_error = task.error or ""
        task.retry_count += 1
        task.status = "queued"
        task.error = None
        task.pending_gate = None
        task.metadata = task.metadata if isinstance(task.metadata, dict) else {}
        task.metadata.pop("execution_checkpoint", None)
        task.metadata.pop("human_blocking_issues", None)
        task.metadata.pop("invalid_workdoc_path", None)
        task.metadata.pop("invalid_workdoc_error", None)
        task.metadata.pop("workdoc_sync_error_type", None)
        task.metadata.pop("workdoc_sync_mode", None)
        task.metadata.pop("workdoc_sync_step", None)
        task.metadata.pop("workdoc_sync_attempt", None)
        guidance = (body.guidance if body else None) or ""
        ts = now_iso()
        if guidance.strip() or previous_error.strip():
            task.metadata["retry_guidance"] = {
                "ts": ts,
                "guidance": guidance.strip(),
                "previous_error": previous_error.strip(),
            }
        if guidance.strip():
            history_list: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
            history_list.append({"action": "retry", "ts": ts, "guidance": guidance.strip(), "previous_error": previous_error.strip()})
        start_from = (body.start_from_step if body else None) or ""
        if start_from.strip():
            task.metadata["retry_from_step"] = start_from.strip()
        elif task.current_step:
            # Only auto-default to steps that exist in the pipeline or
            # the special review/commit phases.  Synthetic steps like
            # "implement_fix" are not in the template and would cause
            # the retry-from logic to silently skip all phase-1 steps.
            valid_restart = set(task.pipeline_template or []) | {"review", "commit"}
            if task.current_step in valid_restart:
                task.metadata["retry_from_step"] = task.current_step
            else:
                task.metadata.pop("retry_from_step", None)
        else:
            task.metadata.pop("retry_from_step", None)
        target_step = _effective_retry_target_step(task)
        if guidance.strip():
            set_active_human_guidance(
                task,
                source="retry",
                guidance=guidance.strip(),
                created_at=ts,
                target_step=target_step,
                fallback_step=_guidance_fallback_step(task, target_step=target_step),
            )
        else:
            clear_active_human_guidance(task, cleared_at=ts)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(
            channel="tasks",
            event_type="task.retry",
            entity_id=task.id,
            payload={
                "retry_count": task.retry_count,
                "start_from_step": task.metadata.get("retry_from_step"),
                "has_guidance": bool(guidance.strip()),
                "guidance": guidance.strip(),
                "previous_error_present": bool(previous_error.strip()),
            },
        )
        return {"task": _task_payload(task, container, orchestrator)}

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Cancel a task and emit a cancellation event.
        
        Args:
            task_id: Identifier of the task to cancel.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the cancelled task.
        
        Raises:
            HTTPException: If the task does not exist.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        try:
            task = orchestrator.cancel_task(task_id, source="api_cancel")
        except ValueError as exc:
            if "Task not found" in str(exc):
                raise HTTPException(status_code=404, detail=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        return {"task": _task_payload(task, container, orchestrator)}

    @router.post("/tasks/{task_id}/approve-gate")
    async def approve_gate(task_id: str, body: ApproveGateRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Clear a pending human gate on a task.
        
        Args:
            task_id: Identifier of the task awaiting gate approval.
            body: Gate approval payload, including optional gate name validation.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with the updated task and cleared gate value.
        
        Raises:
            HTTPException: If the task is missing or no matching pending gate exists.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if not task.pending_gate:
            raise HTTPException(status_code=400, detail="No pending gate on this task")
        if body.gate and body.gate != task.pending_gate:
            raise HTTPException(status_code=400, detail=f"Gate mismatch: pending={task.pending_gate}, requested={body.gate}")
        action = str(getattr(body, "action", "approve") or "approve").strip().lower()
        if action not in {"approve", "request_changes"}:
            raise HTTPException(status_code=400, detail=f"Unsupported gate action: {action}")

        cleared_gate = task.pending_gate
        acted_at = now_iso()
        should_resume = task.status == "in_progress"
        if not isinstance(task.metadata, dict):
            task.metadata = {}

        if action == "request_changes":
            start_from_step = _request_changes_step_for_gate(task, cleared_gate)
            guidance = str(getattr(body, "guidance", "") or "").strip()
            task.retry_count += 1
            task.status = "queued"
            task.error = None
            task.pending_gate = None
            task.current_agent_id = None
            task.metadata.pop("execution_checkpoint", None)
            if start_from_step:
                task.metadata["retry_from_step"] = start_from_step
            else:
                task.metadata.pop("retry_from_step", None)
            task.metadata["requested_changes"] = {
                "ts": acted_at,
                "guidance": guidance,
                "gate": cleared_gate,
            }
            if guidance:
                set_active_human_guidance(
                    task,
                    source="gate_request_changes",
                    guidance=guidance,
                    created_at=acted_at,
                    target_step=start_from_step,
                    fallback_step=_guidance_fallback_step(task, target_step=start_from_step),
                    gate=cleared_gate,
                )
            else:
                clear_active_human_guidance(task, cleared_at=acted_at)
            history: list[dict[str, Any]] = task.metadata.setdefault("human_review_actions", [])
            history.append(
                {
                    "action": "request_changes",
                    "ts": acted_at,
                    "guidance": guidance,
                    "gate": cleared_gate,
                }
            )
            _mark_latest_run_interrupted(
                container,
                task,
                summary=f"Changes requested at gate: {cleared_gate}",
                ts=acted_at,
            )
            task.updated_at = acted_at
            container.tasks.upsert(task)
            bus.emit(
                channel="tasks",
                event_type="task.changes_requested",
                entity_id=task.id,
                payload={"gate": cleared_gate, "guidance": guidance, "start_from_step": start_from_step},
            )
            orchestrator.ensure_worker()
            return {
                "task": _task_payload(task, container, orchestrator),
                "cleared_gate": cleared_gate,
                "message": _gate_changes_requested_message(cleared_gate, start_from_step=start_from_step),
                "approved_at": acted_at,
            }

        if should_resume:
            checkpoint = task.metadata.get("execution_checkpoint")
            checkpoint_payload = dict(checkpoint) if isinstance(checkpoint, dict) else {}
            checkpoint_payload["gate"] = cleared_gate
            checkpoint_payload["resume_step"] = (
                str(checkpoint_payload.get("resume_step") or "").strip() or _resume_step_for_gate(cleared_gate)
            )
            checkpoint_payload["approved_gate"] = cleared_gate
            checkpoint_payload["resume_requested_at"] = acted_at
            task.metadata["execution_checkpoint"] = checkpoint_payload
        task.pending_gate = None
        task.updated_at = acted_at
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.gate_approved", entity_id=task.id, payload={"gate": cleared_gate})
        if should_resume:
            bus.emit(channel="tasks", event_type="task.resume_requested", entity_id=task.id, payload={"gate": cleared_gate})
            orchestrator.ensure_worker()
        return {
            "task": _task_payload(task, container, orchestrator),
            "cleared_gate": cleared_gate,
            "message": _gate_approved_message(cleared_gate, will_resume=should_resume),
            "approved_at": acted_at,
        }

    @router.post("/tasks/{task_id}/dependencies")
    async def add_dependency(task_id: str, body: AddDependencyRequest, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Link a task to another task as a blocker.
        
        Args:
            task_id: Identifier of the blocked task.
            body: Dependency payload identifying the blocker task.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated blocked task.
        
        Raises:
            HTTPException: If either task cannot be found.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        blocker = container.tasks.get(body.depends_on)
        if not task or not blocker:
            raise HTTPException(status_code=404, detail="Task or dependency not found")
        if body.depends_on not in task.blocked_by:
            task.blocked_by.append(body.depends_on)
        if task.id not in blocker.blocks:
            blocker.blocks.append(task.id)
        container.tasks.upsert(task)
        container.tasks.upsert(blocker)
        bus.emit(channel="tasks", event_type="task.dependency_added", entity_id=task.id, payload={"depends_on": body.depends_on})
        return {"task": _task_payload(task, container, orchestrator)}

    @router.delete("/tasks/{task_id}/dependencies/{dep_id}")
    async def remove_dependency(task_id: str, dep_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Remove an existing dependency edge between two tasks.
        
        Args:
            task_id: Identifier of the task being unblocked.
            dep_id: Identifier of the blocker task to unlink.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated task.
        
        Raises:
            HTTPException: If the task cannot be found.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task.blocked_by = [item for item in task.blocked_by if item != dep_id]
        container.tasks.upsert(task)
        blocker = container.tasks.get(dep_id)
        if blocker:
            blocker.blocks = [item for item in blocker.blocks if item != task.id]
            container.tasks.upsert(blocker)
        bus.emit(channel="tasks", event_type="task.dependency_removed", entity_id=task.id, payload={"dep_id": dep_id})
        return {"task": _task_payload(task, container, orchestrator)}

    @router.post("/tasks/analyze-dependencies")
    async def analyze_dependencies(project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Run dependency inference and return inferred edges.
        
        Args:
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing inferred dependency edges across tasks.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        orchestrator._maybe_analyze_dependencies()
        # Collect all inferred edges across tasks
        edges: list[dict[str, str]] = []
        for task in container.tasks.list():
            inferred = task.metadata.get("inferred_deps") if isinstance(task.metadata, dict) else None
            if isinstance(inferred, list):
                for dep in inferred:
                    if isinstance(dep, dict):
                        edges.append({"from": dep.get("from", ""), "to": task.id, "reason": dep.get("reason", "")})
        return {"edges": edges}

    @router.post("/tasks/{task_id}/reset-dep-analysis")
    async def reset_dep_analysis(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Remove dependency-analysis metadata from a task.
        
        Args:
            task_id: Identifier of the task whose analysis metadata is reset.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the updated task.
        
        Raises:
            HTTPException: If the task cannot be found.
        """
        container, bus, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if isinstance(task.metadata, dict):
            # Remove inferred blocked_by entries and corresponding blocks on blockers
            inferred = task.metadata.get("inferred_deps")
            if isinstance(inferred, list):
                inferred_from_ids = {
                    dep_id
                    for dep in inferred
                    if isinstance(dep, dict)
                    for dep_id in [dep.get("from")]
                    if isinstance(dep_id, str) and dep_id
                }
                task.blocked_by = [bid for bid in task.blocked_by if bid not in inferred_from_ids]
                for blocker_id in inferred_from_ids:
                    blocker = container.tasks.get(blocker_id)
                    if blocker:
                        blocker.blocks = [bid for bid in blocker.blocks if bid != task.id]
                        container.tasks.upsert(blocker)
            task.metadata.pop("deps_analyzed", None)
            task.metadata.pop("inferred_deps", None)
        task.updated_at = now_iso()
        container.tasks.upsert(task)
        bus.emit(channel="tasks", event_type="task.dep_analysis_reset", entity_id=task.id, payload={})
        return {"task": _task_payload(task, container, orchestrator)}

    @router.get("/tasks/{task_id}/workdoc")
    async def get_task_workdoc(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the orchestrator work document for a task.
        
        Args:
            task_id: Identifier of the task whose work document is requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The serialized work document payload.
        
        Raises:
            HTTPException: If the task cannot be found.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        orchestrator = deps.resolve_orchestrator(project_dir)
        try:
            return orchestrator.get_workdoc(task_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/tasks/{task_id}/plan")
    async def get_task_plan(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Return the current plan document for a task.
        
        Args:
            task_id: Identifier of the task whose plan is requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            The serialized plan document payload.
        
        Raises:
            HTTPException: If the task is missing or plan retrieval fails validation.
        """
        container, _, _ = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        orchestrator = deps.resolve_orchestrator(project_dir)
        try:
            return orchestrator.get_plan_document(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/tasks/{task_id}/plan/refine")
    async def refine_task_plan(
        task_id: str,
        body: PlanRefineRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Queue a plan-refinement job for a task.
        
        Args:
            task_id: Identifier of the task whose plan is being refined.
            body: Refinement request with feedback, optional instructions, and priority.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the queued refinement job.
        
        Raises:
            HTTPException: If the task is missing, feedback is empty, or queueing fails.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        if not str(body.feedback or "").strip():
            raise HTTPException(status_code=400, detail="feedback is required")
        try:
            job = orchestrator.queue_plan_refine_job(
                task_id=task_id,
                base_revision_id=body.base_revision_id,
                feedback=body.feedback,
                instructions=body.instructions,
                priority=body.priority,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"job": job.to_dict()}

    @router.get("/tasks/{task_id}/plan/jobs")
    async def list_plan_refine_jobs(task_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """List plan-refinement jobs associated with a task.
        
        Args:
            task_id: Identifier of the task whose refinement jobs are requested.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing serialized refinement jobs.
        
        Raises:
            HTTPException: If the task is missing or job lookup fails.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            jobs = orchestrator.list_plan_refine_jobs(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"jobs": [job.to_dict() for job in jobs]}

    @router.get("/tasks/{task_id}/plan/jobs/{job_id}")
    async def get_plan_refine_job(task_id: str, job_id: str, project_dir: Optional[str] = Query(None)) -> dict[str, Any]:
        """Fetch a specific plan-refinement job for a task.
        
        Args:
            task_id: Identifier of the owning task.
            job_id: Identifier of the refinement job.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the requested refinement job.
        
        Raises:
            HTTPException: If the task or job cannot be found.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            job = orchestrator.get_plan_refine_job(task_id, job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"job": job.to_dict()}

    @router.post("/tasks/{task_id}/plan/commit")
    async def commit_plan_revision(
        task_id: str,
        body: CommitPlanRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Commit a plan revision as the task's accepted plan.
        
        Args:
            task_id: Identifier of the task whose plan is being committed.
            body: Commit payload containing the revision identifier.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with committed and latest revision identifiers.
        
        Raises:
            HTTPException: If the task is missing or revision commit fails.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        try:
            committed_revision_id = orchestrator.commit_plan_revision(task_id, body.revision_id)
            plan_doc = orchestrator.get_plan_document(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "committed_revision_id": committed_revision_id,
            "latest_revision_id": plan_doc.get("latest_revision_id"),
            "committed_plan_revision_id": plan_doc.get("committed_revision_id"),
        }

    @router.post("/tasks/{task_id}/plan/revisions")
    async def create_plan_revision(
        task_id: str,
        body: CreatePlanRevisionRequest,
        project_dir: Optional[str] = Query(None),
    ) -> dict[str, Any]:
        """Create a manual plan revision for a task.
        
        Args:
            task_id: Identifier of the task whose plan revision is created.
            body: Revision payload containing content and optional parent metadata.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload containing the created plan revision.
        
        Raises:
            HTTPException: If the task is missing or content is invalid.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        if not container.tasks.get(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        if not str(body.content or "").strip():
            raise HTTPException(status_code=400, detail="content is required")
        try:
            revision = orchestrator.create_plan_revision(
                task_id=task_id,
                content=body.content,
                source="human_edit",
                parent_revision_id=body.parent_revision_id,
                feedback_note=body.feedback_note,
                step=None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"revision": revision.to_dict()}

    @router.post("/tasks/{task_id}/generate-tasks")
    async def generate_tasks_from_plan(
        task_id: str, body: GenerateTasksRequest, project_dir: Optional[str] = Query(None)
    ) -> dict[str, Any]:
        """Generate child tasks from selected or override plan text.
        
        Args:
            task_id: Identifier of the parent task receiving generated children.
            body: Generation payload selecting plan source and dependency inference mode.
            project_dir: Optional project directory used to resolve runtime state.
        
        Returns:
            A payload with generated child identifiers and resolved source metadata.
        
        Raises:
            HTTPException: If the task is missing or generation inputs are inconsistent.
        """
        container, _, orchestrator = deps.ctx(project_dir)
        task = container.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        source = body.source
        if source is None:
            # Backward compatibility: previous API accepted only optional plan_override.
            source = "override" if str(body.plan_override or "").strip() else "latest"
        assert source is not None
        if source == "revision" and not body.revision_id:
            raise HTTPException(status_code=400, detail="revision_id is required when source=revision")
        if source == "override" and not str(body.plan_override or "").strip():
            raise HTTPException(status_code=400, detail="plan_override is required when source=override")
        if source != "revision" and body.revision_id:
            raise HTTPException(status_code=400, detail="revision_id is only valid when source=revision")
        if source != "override" and str(body.plan_override or "").strip():
            raise HTTPException(status_code=400, detail="plan_override is only valid when source=override")
        try:
            plan_text, resolved_revision_id = orchestrator.resolve_plan_text_for_generation(
                task_id=task_id,
                source=source,
                revision_id=body.revision_id,
                plan_override=body.plan_override,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        try:
            created_ids = orchestrator.generate_tasks_from_plan(task_id, plan_text, infer_deps=body.infer_deps)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Re-fetch parent to get updated children_ids
        updated_task = container.tasks.get(task_id)
        children: list[dict[str, Any]] = []
        for child_id in created_ids:
            child_task = container.tasks.get(child_id)
            if child_task is not None:
                children.append(_task_payload(child_task, orchestrator=orchestrator))
        return {
            "task": _task_payload(updated_task, orchestrator=orchestrator) if updated_task else None,
            "created_task_ids": created_ids,
            "children": children,
            "source": source,
            "source_revision_id": resolved_revision_id,
        }
