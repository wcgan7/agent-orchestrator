"""Runtime invariant checks and deterministic repair actions."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from ..domain.models import RunRecord, Task, now_iso

if TYPE_CHECKING:
    from .service import OrchestratorService


def _branch_exists(project_dir: Any, branch_name: str) -> bool:
    normalized = str(branch_name or "").strip()
    if not normalized:
        return False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{normalized}"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return False
    return result.returncode == 0


def _clear_cancelled_gate_artifacts(task: Task) -> bool:
    """Remove approval/wait-state artifacts that should not survive cancellation."""
    changed = False
    if task.pending_gate:
        task.pending_gate = None
        changed = True
    if isinstance(task.metadata, dict):
        for key in (
            "pending_precommit_approval",
            "review_stage",
            "execution_checkpoint",
            "requested_changes",
        ):
            if key in task.metadata:
                task.metadata.pop(key, None)
                changed = True
    return changed


def apply_runtime_invariants(
    service: "OrchestratorService",
    *,
    active_future_task_ids: set[str],
    source: str,
) -> dict[str, Any]:
    """Apply strict runtime invariants and return a repair summary."""
    container = service.container
    bus = service.bus
    repairs: list[dict[str, Any]] = []
    now_stamp = now_iso()
    git_backed = (container.project_dir / ".git").exists()

    tasks = container.tasks.list()
    task_by_id = {task.id: task for task in tasks}
    terminal = {"done", "cancelled"}

    def _record(task: Task, *, code: str, message: str) -> None:
        repairs.append({"entity": "task", "task_id": task.id, "code": code, "message": message})
        bus.emit(
            channel="tasks",
            event_type="task.reconciled",
            entity_id=task.id,
            payload={"code": code, "message": message, "source": source},
        )

    for task in tasks:
        changed = False
        metadata = task.metadata if isinstance(task.metadata, dict) else {}

        if task.status == "cancelled":
            if _clear_cancelled_gate_artifacts(task):
                changed = True
                _record(
                    task,
                    code="cancelled_cleanup",
                    message="Cleared gate and review artifacts from cancelled task.",
                )
            cleanup = service._cleanup_cancelled_task_context(
                task,
                active_future_task_ids=active_future_task_ids,
            )
            if cleanup.get("deferred"):
                if cleanup.get("metadata_changed"):
                    changed = True
                    _record(
                        task,
                        code="cancelled_cleanup_deferred",
                        message="Deferred cancelled context cleanup because execution is still active.",
                    )
            elif any(
                bool(cleanup.get(key))
                for key in ("metadata_changed", "worktree_removed", "branch_deleted", "lease_released")
            ):
                changed = True
                _record(
                    task,
                    code="cancelled_context_cleanup",
                    message="Cleaned retained context and branch artifacts from cancelled task.",
                )

        if task.status == "queued" and task.pending_gate:
            gate_name = str(task.pending_gate or "").strip() or "approval"
            task.status = "blocked"
            task.error = f"Invalid queued gate state detected ({gate_name})."
            task.pending_gate = None
            if isinstance(task.metadata, dict):
                task.metadata.pop("execution_checkpoint", None)
            changed = True
            _record(
                task,
                code="queued_with_pending_gate",
                message=f"Queued task carried pending gate '{gate_name}' and was blocked.",
            )

        if task.status == "in_review" and git_backed:
            is_precommit = bool(metadata.get("pending_precommit_approval"))
            if is_precommit:
                preserved_branch = str(metadata.get("preserved_branch") or "").strip()
                if not preserved_branch or not _branch_exists(container.project_dir, preserved_branch):
                    task.status = "blocked"
                    task.error = "Pre-commit context missing; request changes to regenerate implementation context."
                    task.pending_gate = None
                    task.current_agent_id = None
                    metadata.pop("pending_precommit_approval", None)
                    metadata.pop("review_stage", None)
                    changed = True
                    _record(
                        task,
                        code="precommit_context_missing",
                        message="Pre-commit review context was missing and task was blocked.",
                    )

        if task.status == "blocked" and git_backed:
            context_raw = metadata.get("task_context")
            context = context_raw if isinstance(context_raw, dict) else {}
            expected_on_retry = bool(context.get("expected_on_retry"))
            if expected_on_retry:
                has_retained = service._resolve_retained_task_worktree(task) is not None
                preserved_branch = str(metadata.get("preserved_branch") or "").strip()
                has_preserved = bool(preserved_branch and _branch_exists(container.project_dir, preserved_branch))
                if not has_retained and not has_preserved:
                    message = "Retry context missing; request changes to regenerate task context."
                    if task.error != message:
                        task.error = message
                        changed = True
                    _record(
                        task,
                        code="context_missing_for_retry",
                        message="Blocked task expected retry context but retained/preserved references were missing.",
                    )

        if task.status == "in_progress" and not task.pending_gate:
            has_active_future = task.id in active_future_task_ids
            resume_requested = False
            checkpoint = metadata.get("execution_checkpoint")
            if isinstance(checkpoint, dict):
                resume_requested = bool(str(checkpoint.get("resume_requested_at") or "").strip())
            has_active_lease = service._execution_lease_active(task)
            if not has_active_future and not has_active_lease and not resume_requested:
                task.status = "queued"
                task.current_agent_id = None
                task.current_step = None
                task.pending_gate = None
                task.error = "Recovered from stale in_progress state"
                if isinstance(task.metadata, dict):
                    task.metadata.pop("pipeline_phase", None)
                changed = True
                _record(
                    task,
                    code="stale_in_progress",
                    message="Task had no future/lease and was requeued.",
                )

        if changed:
            container.tasks.upsert(task)

    # Ensure terminal tasks do not keep active runs.
    task_terminal_status = {
        task.id: task.status
        for task in container.tasks.list()
        if task.status in terminal
    }
    runs = container.runs.list()
    for run in runs:
        owner_status = task_terminal_status.get(run.task_id)
        if owner_status is None:
            continue
        if run.status not in {"in_progress", "waiting_gate"}:
            continue
        run.status = "cancelled" if owner_status == "cancelled" else "done"
        run.finished_at = run.finished_at or now_stamp
        if not run.summary:
            run.summary = "Run finalized by invariant reconciler."
        container.runs.upsert(run)
        repairs.append(
            {
                "entity": "run",
                "task_id": run.task_id,
                "run_id": run.id,
                "code": "terminal_run_cleanup",
                "message": "Terminal task had active run metadata; run finalized.",
            }
        )

    return {
        "source": source,
        "checked_tasks": len(tasks),
        "repairs": len(repairs),
        "items": repairs,
    }
