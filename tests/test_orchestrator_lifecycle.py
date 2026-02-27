from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from agent_orchestrator.server.api import create_app
from agent_orchestrator.runtime.domain.models import RunRecord, Task, now_iso
from agent_orchestrator.runtime.events.bus import EventBus
from agent_orchestrator.runtime.orchestrator.service import OrchestratorService
from agent_orchestrator.runtime.orchestrator.worker_adapter import DefaultWorkerAdapter
from agent_orchestrator.runtime.storage.container import Container


def test_app_shutdown_stops_orchestrators(tmp_path: Path) -> None:
    app = create_app(project_dir=tmp_path, worker_adapter=DefaultWorkerAdapter())
    orchestrator: OrchestratorService | None = None
    with TestClient(app) as client:
        response = client.get("/api/orchestrator/status")
        assert response.status_code == 200
        values = list(app.state.orchestrators.values())
        assert values
        orchestrator = values[0]
        assert orchestrator._thread is not None
        assert orchestrator._thread.is_alive()

    assert orchestrator is not None
    assert app.state.orchestrators == {}
    assert orchestrator._thread is None


def test_in_progress_tasks_recover_on_worker_start(tmp_path: Path) -> None:
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)

    task = Task(
        title="Recover me",
        task_type="feature",
        status="in_progress",
    )
    container.tasks.upsert(task)
    run = RunRecord(task_id=task.id, status="in_progress", started_at=now_iso())
    container.runs.upsert(run)

    service = OrchestratorService(container, bus, worker_adapter=DefaultWorkerAdapter())
    service.ensure_worker()
    try:
        recovered_task = container.tasks.get(task.id)
        assert recovered_task is not None
        assert recovered_task.status == "queued"
        assert recovered_task.error == "Recovered from interrupted run"

        recovered_run = container.runs.list()[0]
        assert recovered_run.status == "interrupted"
        assert recovered_run.finished_at is not None
    finally:
        service.shutdown()


def test_resume_reattaches_scheduler_after_shutdown(tmp_path: Path) -> None:
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=DefaultWorkerAdapter())
    service.ensure_worker()
    try:
        assert service.status()["scheduler_attached"] is True
        service.shutdown(timeout=1.0)
        assert service.status()["scheduler_attached"] is False

        resumed = service.control("resume")
        assert resumed["status"] == "running"
        assert resumed["scheduler_attached"] is True
    finally:
        service.shutdown()


def test_reset_action_reattaches_scheduler_without_status_change(tmp_path: Path) -> None:
    container = Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus, worker_adapter=DefaultWorkerAdapter())
    service.ensure_worker()
    try:
        service.control("pause")
        service.shutdown(timeout=1.0)
        assert service.status()["scheduler_attached"] is False

        reset_status = service.control("reset")
        assert reset_status["status"] == "paused"
        assert reset_status["scheduler_attached"] is True
    finally:
        # Give the loop a beat to observe paused config before shutdown.
        time.sleep(0.05)
        service.shutdown()
