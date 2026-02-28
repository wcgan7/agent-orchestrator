from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from agent_orchestrator.runtime.domain.models import Task
from agent_orchestrator.runtime.events import EventBus
from agent_orchestrator.runtime.orchestrator.service import OrchestratorService
from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter, StepResult
from agent_orchestrator.runtime.storage.container import Container
from agent_orchestrator.server.api import create_app
from tests.critical.assertions import assert_task_lifecycle_invariants


def _service(tmp_path: Path, *, worker_adapter: object | None = None) -> tuple[Container, OrchestratorService]:
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=worker_adapter or DefaultWorkerAdapter())
    return container, service


def test_manual_run_waits_for_scheduler_claim_and_returns_terminal(tmp_path: Path) -> None:
    """run_task should block on already-claimed scheduler work and return final task state."""

    step_started = threading.Event()
    release_step = threading.Event()

    class BlockingAdapter(DefaultWorkerAdapter):
        blocked_once = False

        def run_step(self, *, task: Task, step: str, attempt: int) -> StepResult:
            if not self.blocked_once:
                self.blocked_once = True
                step_started.set()
                release_step.wait(timeout=10)
            return super().run_step(task=task, step=step, attempt=attempt)

    container, service = _service(tmp_path, worker_adapter=BlockingAdapter())
    task = Task(title="Race wait task", task_type="chore", status="queued", hitl_mode="autopilot")
    container.tasks.upsert(task)
    service.ensure_worker()
    try:
        assert step_started.wait(timeout=5), "scheduler did not claim task in time"

        out: dict[str, Task] = {}
        done = threading.Event()

        def _run() -> None:
            out["task"] = service.run_task(task.id)
            done.set()

        runner = threading.Thread(target=_run, daemon=True)
        runner.start()
        time.sleep(0.2)
        assert not done.is_set(), "run_task returned before scheduler-owned step was released"

        release_step.set()
        runner.join(timeout=10)
        assert done.is_set(), "run_task did not finish after scheduler execution completed"
        result = out["task"]
        assert result.status == "done"
        assert_task_lifecycle_invariants(result, service=service)
    finally:
        service.shutdown()


def test_request_changes_rerun_under_active_scheduler_keeps_valid_state(tmp_path: Path) -> None:
    """Gate request-changes + immediate rerun should never surface missing retry-context failures."""

    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    with TestClient(app) as client:
        created = client.post(
            "/api/tasks",
            json={
                "title": "Interleaving request changes",
                "task_type": "feature",
                "hitl_mode": "supervised",
                "status": "queued",
                "pipeline_template": ["plan", "implement", "verify", "review", "commit"],
            },
        )
        assert created.status_code == 200
        task = created.json()["task"]

        first_run = client.post(f"/api/tasks/{task['id']}/run")
        assert first_run.status_code == 200
        parked = first_run.json()["task"]
        assert parked["status"] == "in_progress"
        assert parked["pending_gate"] == "before_implement"

        request_changes = client.post(
            f"/api/tasks/{task['id']}/approve-gate",
            json={
                "gate": "before_implement",
                "action": "request_changes",
                "guidance": "narrow the implementation plan",
            },
        )
        assert request_changes.status_code == 200

        # approve-gate(request_changes) starts scheduler work; rerun immediately to
        # exercise interleavings between manual run and background claim.
        rerun = client.post(f"/api/tasks/{task['id']}/run")
        assert rerun.status_code == 200
        rerun_task = rerun.json()["task"]
        error_text = str(rerun_task.get("error") or "")
        assert "Retained task context is missing" not in error_text
        assert "Retry context missing" not in error_text

        orchestrators = list(app.state.orchestrators.values())
        assert orchestrators, "expected active orchestrator cache"
        service = orchestrators[0]
        latest = service.container.tasks.get(task["id"])
        assert latest is not None
        assert_task_lifecycle_invariants(latest, service=service)
