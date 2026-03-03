"""Tests for PlanManager.is_plan_mutable() and plan-lock guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.events.bus import EventBus
from agent_orchestrator.runtime.orchestrator import OrchestratorService, DefaultWorkerAdapter
from agent_orchestrator.runtime.storage.bootstrap import ensure_state_root
from agent_orchestrator.runtime.storage.container import Container


def _setup(tmp_path: Path) -> tuple[Container, OrchestratorService]:
    ensure_state_root(tmp_path)
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=DefaultWorkerAdapter())
    return container, service


# -- is_plan_mutable ----------------------------------------------------------


def test_mutable_when_backlog(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="backlog")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


def test_mutable_when_queued(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="queued")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


def test_locked_when_done(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="done")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is False


def test_locked_when_cancelled(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="cancelled")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is False


def test_locked_when_in_review(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="in_review")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is False


def test_mutable_at_before_plan_gate(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="in_progress", pending_gate="before_plan")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


def test_mutable_at_before_implement_gate(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="in_progress", pending_gate="before_implement")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


def test_mutable_at_before_generate_tasks_gate(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="in_progress", pending_gate="before_generate_tasks")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


def test_mutable_when_current_step_is_plan(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="plan",
        pipeline_template=["plan", "implement", "verify", "review", "commit"],
    )
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


def test_locked_when_current_step_past_plan(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="implement",
        pipeline_template=["plan", "implement", "verify", "review", "commit"],
    )
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is False


def test_locked_when_current_step_is_verify(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="verify",
        pipeline_template=["plan", "implement", "verify", "review", "commit"],
    )
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is False


def test_locked_when_pipeline_has_no_plan_step(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="bug_fix", status="in_progress",
        current_step="implement",
        pipeline_template=["implement", "verify", "commit"],
    )
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is False


def test_locked_for_virtual_step(tmp_path: Path) -> None:
    """Virtual steps like implement_fix are not in the pipeline list — treat as past plan."""
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="implement_fix",
        pipeline_template=["plan", "implement", "verify", "review", "commit"],
    )
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is False


def test_mutable_when_step_before_plan_in_pipeline(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="refactor", status="in_progress",
        current_step="analyze",
        pipeline_template=["analyze", "plan", "implement", "verify", "commit"],
    )
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


def test_mutable_missing_task_returns_false(tmp_path: Path) -> None:
    _, service = _setup(tmp_path)
    assert service._plan_manager.is_plan_mutable("nonexistent") is False


def test_mutable_permissive_when_no_pipeline_or_step(tmp_path: Path) -> None:
    """When pipeline_template or current_step is empty, be permissive."""
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="in_progress")
    container.tasks.upsert(task)
    assert service._plan_manager.is_plan_mutable(task.id) is True


# -- Guard: create_plan_revision (human_edit) ---------------------------------


def test_create_revision_human_edit_blocked_when_locked(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="implement",
        pipeline_template=["plan", "implement", "verify", "commit"],
    )
    container.tasks.upsert(task)
    with pytest.raises(RuntimeError, match="Plan is locked"):
        service._plan_manager.create_plan_revision(
            task_id=task.id, content="new plan", source="human_edit",
        )


def test_create_revision_worker_plan_allowed_when_locked(tmp_path: Path) -> None:
    """Worker-generated revisions bypass the lock so pipelines can still write."""
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="implement",
        pipeline_template=["plan", "implement", "verify", "commit"],
    )
    container.tasks.upsert(task)
    rev = service._plan_manager.create_plan_revision(
        task_id=task.id, content="worker plan", source="worker_plan",
    )
    assert rev.content == "worker plan"


# -- Guard: commit_plan_revision -----------------------------------------------


def test_commit_revision_blocked_when_locked(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="implement",
        pipeline_template=["plan", "implement", "verify", "commit"],
    )
    container.tasks.upsert(task)
    # Create a revision via worker (allowed)
    rev = service._plan_manager.create_plan_revision(
        task_id=task.id, content="plan text", source="worker_plan",
    )
    with pytest.raises(RuntimeError, match="Plan is locked"):
        service._plan_manager.commit_plan_revision(task.id, rev.id)


# -- Guard: queue_plan_refine_job ----------------------------------------------


def test_queue_refine_blocked_when_locked(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(
        title="T", task_type="feature", status="in_progress",
        current_step="implement",
        pipeline_template=["plan", "implement", "verify", "commit"],
    )
    container.tasks.upsert(task)
    # Need at least one revision for the refine path
    service._plan_manager.create_plan_revision(
        task_id=task.id, content="plan text", source="worker_plan",
    )
    with pytest.raises(RuntimeError, match="Plan is locked"):
        service._plan_manager.queue_plan_refine_job(task_id=task.id, feedback="tweak it")


# -- get_plan_document exposes plan_mutable ------------------------------------


def test_plan_document_includes_plan_mutable(tmp_path: Path) -> None:
    container, service = _setup(tmp_path)
    task = Task(title="T", task_type="feature", status="queued")
    container.tasks.upsert(task)
    doc = service._plan_manager.get_plan_document(task.id)
    assert doc["plan_mutable"] is True

    # Move to implement → locked
    task.status = "in_progress"  # type: ignore[assignment]
    task.current_step = "implement"
    task.pipeline_template = ["plan", "implement", "verify", "commit"]
    container.tasks.upsert(task)
    doc = service._plan_manager.get_plan_document(task.id)
    assert doc["plan_mutable"] is False
