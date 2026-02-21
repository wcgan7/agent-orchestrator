"""Task/execution helper utilities for runtime API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

from ..domain.models import Task
from .helpers import _normalize_human_blocking_issues, _priority_rank

if TYPE_CHECKING:
    from ..storage.container import Container


def _iso_delta_seconds(start: str, end: str) -> Optional[float]:
    """Compute elapsed seconds between two ISO-8601 timestamps."""
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max((e - s).total_seconds(), 0.0)
    except (ValueError, TypeError):
        return None


def _build_execution_summary(task: Task, container: "Container") -> Optional[dict[str, Any]]:
    """Build execution summary from the latest RunRecord's step data."""
    if task.status not in ("in_review", "blocked", "done"):
        return None
    if not task.run_ids:
        return None
    run = None
    for run_id in reversed(task.run_ids):
        run = container.runs.get(run_id)
        if run:
            break
    if not run or not run.steps:
        return None
    steps: list[dict[str, Any]] = []
    for step_data in run.steps:
        if not isinstance(step_data, dict):
            continue
        step_status = str(step_data.get("status") or "unknown")
        if step_status == "skipped":
            continue
        entry: dict[str, Any] = {
            "step": str(step_data.get("step") or "unknown"),
            "status": step_status,
            "summary": str(step_data.get("summary") or ""),
        }
        if step_data.get("open_counts") and isinstance(step_data["open_counts"], dict):
            entry["open_counts"] = step_data["open_counts"]
        if step_data.get("commit"):
            entry["commit"] = str(step_data["commit"])
        started_at_raw = step_data.get("started_at")
        ts_raw = step_data.get("ts")
        if isinstance(started_at_raw, str) and isinstance(ts_raw, str):
            entry["duration_seconds"] = _iso_delta_seconds(started_at_raw, ts_raw)
        steps.append(entry)
    if not steps:
        return None
    status_labels = {
        "in_review": "Awaiting human review",
        "blocked": "Blocked",
        "done": "Completed",
        "error": "Failed",
        "running": "Running",
    }
    run_summary = status_labels.get(run.status) or status_labels.get(task.status) or run.status
    run_duration = _iso_delta_seconds(run.started_at, run.finished_at) if run.started_at and run.finished_at else None
    return {
        "run_id": run.id,
        "run_status": run.status,
        "run_summary": run_summary,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "duration_seconds": run_duration,
        "steps": steps,
    }


def _task_payload(task: Task, container: Optional["Container"] = None) -> dict[str, Any]:
    payload = task.to_dict()
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    payload["human_blocking_issues"] = _normalize_human_blocking_issues(metadata.get("human_blocking_issues"))
    raw_actions = metadata.get("human_review_actions")
    payload["human_review_actions"] = list(raw_actions) if isinstance(raw_actions, list) else []
    if container is not None:
        summary = _build_execution_summary(task, container)
        if summary is not None:
            payload["execution_summary"] = summary
    return payload


def _execution_batches(tasks: list[Task]) -> list[list[str]]:
    by_id = {task.id: task for task in tasks}
    indegree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {task.id: [] for task in tasks}
    for task in tasks:
        refs = [dep_id for dep_id in task.blocked_by if dep_id in by_id]
        indegree[task.id] = len(refs)
        for dep_id in refs:
            dependents.setdefault(dep_id, []).append(task.id)

    ready = sorted(
        [task_id for task_id, degree in indegree.items() if degree == 0],
        key=lambda tid: (_priority_rank(by_id[tid].priority), by_id[tid].created_at),
    )
    batches: list[list[str]] = []
    while ready:
        batch = list(ready)
        batches.append(batch)
        next_ready: list[str] = []
        for task_id in batch:
            for dep_id in dependents.get(task_id, []):
                indegree[dep_id] -= 1
                if indegree[dep_id] == 0:
                    next_ready.append(dep_id)
        ready = sorted(next_ready, key=lambda tid: (_priority_rank(by_id[tid].priority), by_id[tid].created_at))
    return batches


def _has_unresolved_blockers(container: "Container", task: Task) -> Optional[str]:
    for dep_id in task.blocked_by:
        dep = container.tasks.get(dep_id)
        if dep is None or dep.status not in {"done", "cancelled"}:
            return dep_id
    return None
