"""Shared invariant assertions for critical task lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_orchestrator.runtime.domain.models import Task


def _task_metadata(task: Task) -> dict[str, Any]:
    return task.metadata if isinstance(task.metadata, dict) else {}


def _task_context(task: Task) -> dict[str, Any]:
    raw = _task_metadata(task).get("task_context")
    return raw if isinstance(raw, dict) else {}


def _path_exists(path_text: str) -> bool:
    if not path_text.strip():
        return False
    try:
        return Path(path_text).expanduser().exists()
    except Exception:
        return False


def assert_task_lifecycle_invariants(task: Task, *, service: Any | None = None) -> None:
    """Assert core runtime invariants used by critical pipeline tests.

    Args:
        task: Task under validation.
        service: Optional orchestrator service used for retained-context resolution.
    """
    metadata = _task_metadata(task)
    context = _task_context(task)
    expected_on_retry = bool(context.get("expected_on_retry"))
    pending_precommit = bool(metadata.get("pending_precommit_approval"))
    preserved_branch = str(metadata.get("preserved_branch") or "").strip()
    worktree_dir = str(metadata.get("worktree_dir") or "").strip()
    context_worktree_dir = str(context.get("worktree_dir") or "").strip()

    # Queued tasks should not be parked at a gate.
    if task.status == "queued":
        assert task.pending_gate is None, f"queued task {task.id} unexpectedly has pending_gate={task.pending_gate!r}"

    # Pre-commit review state must carry preserved branch context.
    if task.status == "in_review" and pending_precommit:
        assert preserved_branch, f"in_review pre-commit task {task.id} is missing preserved_branch"

    # If retry context is expected, either retained/preserved context resolves, or
    # the task is explicitly fail-closed with the expected retry-context error.
    if task.status == "blocked" and expected_on_retry:
        has_preserved = bool(preserved_branch)
        if service is not None:
            has_retained = service._resolve_retained_task_worktree(task) is not None
        else:
            candidate = context_worktree_dir or worktree_dir
            has_retained = _path_exists(candidate)
        if not has_retained and not has_preserved:
            error_text = str(task.error or "")
            assert (
                "Retry context missing" in error_text
                or "Retained task context is missing" in error_text
            ), f"blocked task {task.id} expects retry context but is not marked fail-closed"

    # Non-retained queued/in-progress tasks should not carry dangling context pointers.
    if task.status in {"queued", "in_progress", "in_review"} and not expected_on_retry:
        candidate = context_worktree_dir or worktree_dir
        if candidate:
            assert _path_exists(candidate), (
                f"non-retained task {task.id} carries stale worktree pointer: {candidate}"
            )
