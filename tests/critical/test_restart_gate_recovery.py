from __future__ import annotations

import subprocess
from pathlib import Path

from agent_orchestrator.runtime.domain.models import RunRecord, Task, now_iso
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator.service import OrchestratorService
from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter
from agent_orchestrator.runtime.storage.container import Container
from tests.critical.assertions import assert_task_lifecycle_invariants


def _service(tmp_path: Path) -> tuple[Container, OrchestratorService]:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=DefaultWorkerAdapter())
    return container, service


def test_waiting_gate_task_survives_restart_reconcile(tmp_path: Path) -> None:
    """in_progress gate-waiting tasks should remain stable across restart+reconcile."""
    container, service = _service(tmp_path)
    task = Task(
        title="Restart gate waiting",
        task_type="feature",
        status="in_progress",
        pending_gate="before_implement",
        current_step="plan",
        metadata={"execution_checkpoint": {"gate": "before_implement", "resume_step": "implement"}},
    )
    container.tasks.upsert(task)
    run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso())
    container.runs.upsert(run)

    service.ensure_worker()
    try:
        recovered = container.tasks.get(task.id)
        assert recovered is not None
        assert recovered.status == "in_progress"
        assert recovered.pending_gate == "before_implement"
        assert recovered.current_agent_id is None
        assert_task_lifecycle_invariants(recovered, service=service)

        updated = service.reconcile(source="manual")
        assert isinstance(updated, dict)
        after = container.tasks.get(task.id)
        assert after is not None
        assert after.status == "in_progress"
        assert after.pending_gate == "before_implement"
        assert_task_lifecycle_invariants(after, service=service)
    finally:
        service.shutdown()


def test_reconcile_marks_missing_expected_retry_context_fail_closed(tmp_path: Path) -> None:
    """Blocked tasks that expect retry context but lost it must stay fail-closed after reconcile."""
    container, service = _service(tmp_path)
    missing_wt = container.state_root / "worktrees" / "missing-retry-context"
    task = Task(
        id="missing-retry-context",
        title="Missing retry context",
        task_type="feature",
        status="blocked",
        metadata={
            "worktree_dir": str(missing_wt),
            "task_context": {
                "context_id": "ctx-missing-retry",
                "worktree_dir": str(missing_wt),
                "task_branch": "task-missing-retry-context",
                "retained": True,
                "expected_on_retry": True,
            },
        },
    )
    container.tasks.upsert(task)

    summary = service.reconcile(source="manual")
    updated = container.tasks.get(task.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert "Retry context missing" in str(updated.error or "")
    assert any(
        item.get("task_id") == task.id and item.get("code") == "context_missing_for_retry"
        for item in (summary.get("items") or [])
    )
    assert_task_lifecycle_invariants(updated, service=service)
