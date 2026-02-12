"""Tests for HITL mode gate enforcement in the orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from feature_prd_runner.v3.domain.models import Task
from feature_prd_runner.v3.events import EventBus
from feature_prd_runner.v3.orchestrator import OrchestratorService
from feature_prd_runner.v3.storage.container import V3Container


def _service(tmp_path: Path) -> tuple[V3Container, OrchestratorService, EventBus]:
    container = V3Container(tmp_path)
    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)
    return container, service, bus


# ---------------------------------------------------------------------------
# Autopilot — no gates
# ---------------------------------------------------------------------------


def test_autopilot_no_gates(tmp_path: Path) -> None:
    """Task with hitl_mode='autopilot' runs straight to done, no gates."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Auto task",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="autopilot",
    )
    container.tasks.upsert(task)

    result = service.run_task(task.id)

    assert result.status == "done"
    assert result.pending_gate is None


# ---------------------------------------------------------------------------
# Supervised — blocks at every gate
# ---------------------------------------------------------------------------


def test_supervised_blocks_at_gates(tmp_path: Path) -> None:
    """Supervised mode activates before_plan, before_implement, before_commit."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Supervised task",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="supervised",
    )
    container.tasks.upsert(task)

    gates_seen: list[str] = []

    def _mock_wait(t: Task, gate_name: str, timeout: int = 3600) -> bool:
        gates_seen.append(gate_name)
        # Simulate instant approval by clearing the gate
        t.pending_gate = None
        container.tasks.upsert(t)
        return True

    with patch.object(service, "_wait_for_gate", side_effect=_mock_wait):
        result = service.run_task(task.id)

    assert result.status == "done"
    assert "before_plan" in gates_seen
    assert "before_implement" in gates_seen
    assert "before_commit" in gates_seen


# ---------------------------------------------------------------------------
# Review-only — gates after_implement and before_commit only
# ---------------------------------------------------------------------------


def test_review_only_gates_after_implement_and_before_commit(tmp_path: Path) -> None:
    """review_only mode only activates after_implement and before_commit."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Review-only task",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="review_only",
    )
    container.tasks.upsert(task)

    gates_seen: list[str] = []

    def _mock_wait(t: Task, gate_name: str, timeout: int = 3600) -> bool:
        gates_seen.append(gate_name)
        t.pending_gate = None
        container.tasks.upsert(t)
        return True

    with patch.object(service, "_wait_for_gate", side_effect=_mock_wait):
        result = service.run_task(task.id)

    assert result.status == "done"
    assert "before_plan" not in gates_seen
    assert "before_implement" not in gates_seen
    assert "after_implement" in gates_seen
    assert "before_commit" in gates_seen


# ---------------------------------------------------------------------------
# Collaborative — gates after_implement and before_commit
# ---------------------------------------------------------------------------


def test_collaborative_gates(tmp_path: Path) -> None:
    """collaborative mode activates after_implement and before_commit."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Collaborative task",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="collaborative",
    )
    container.tasks.upsert(task)

    gates_seen: list[str] = []

    def _mock_wait(t: Task, gate_name: str, timeout: int = 3600) -> bool:
        gates_seen.append(gate_name)
        t.pending_gate = None
        container.tasks.upsert(t)
        return True

    with patch.object(service, "_wait_for_gate", side_effect=_mock_wait):
        result = service.run_task(task.id)

    assert result.status == "done"
    assert "before_plan" not in gates_seen
    assert "after_implement" in gates_seen
    assert "before_commit" in gates_seen


# ---------------------------------------------------------------------------
# Gate timeout — blocks the task
# ---------------------------------------------------------------------------


def test_gate_timeout_blocks_task(tmp_path: Path) -> None:
    """If nobody approves the gate, _wait_for_gate returns False → blocked."""
    container, service, _ = _service(tmp_path)
    task = Task(
        title="Timeout task",
        status="ready",
        approval_mode="auto_approve",
        hitl_mode="supervised",
    )
    container.tasks.upsert(task)

    def _mock_wait_reject(t: Task, gate_name: str, timeout: int = 3600) -> bool:
        # Simulate gate not being approved
        return False

    with patch.object(service, "_wait_for_gate", side_effect=_mock_wait_reject):
        result = service.run_task(task.id)

    assert result.status == "blocked"
    assert "gate" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Gate approve API endpoint
# ---------------------------------------------------------------------------


def test_gate_approve_api(tmp_path: Path) -> None:
    """POST approve-gate clears pending_gate."""
    from fastapi.testclient import TestClient

    from feature_prd_runner.v3.api.router import create_v3_router

    container = V3Container(tmp_path)

    def resolve_container(_: Any = None) -> V3Container:
        return container

    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    def resolve_orchestrator(_: Any = None) -> OrchestratorService:
        return service

    router = create_v3_router(resolve_container, resolve_orchestrator, {})

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    task = Task(title="Gate test", status="in_progress", pending_gate="before_plan")
    container.tasks.upsert(task)

    resp = client.post(f"/api/v3/tasks/{task.id}/approve-gate", json={"gate": "before_plan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cleared_gate"] == "before_plan"
    assert data["task"]["pending_gate"] is None


def test_gate_approve_api_no_pending(tmp_path: Path) -> None:
    """400 when no pending gate on the task."""
    from fastapi.testclient import TestClient

    from feature_prd_runner.v3.api.router import create_v3_router

    container = V3Container(tmp_path)

    def resolve_container(_: Any = None) -> V3Container:
        return container

    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    def resolve_orchestrator(_: Any = None) -> OrchestratorService:
        return service

    router = create_v3_router(resolve_container, resolve_orchestrator, {})

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    task = Task(title="No gate task", status="in_progress", pending_gate=None)
    container.tasks.upsert(task)

    resp = client.post(f"/api/v3/tasks/{task.id}/approve-gate", json={})
    assert resp.status_code == 400


def test_gate_approve_api_mismatch(tmp_path: Path) -> None:
    """400 when gate name doesn't match the pending gate."""
    from fastapi.testclient import TestClient

    from feature_prd_runner.v3.api.router import create_v3_router

    container = V3Container(tmp_path)

    def resolve_container(_: Any = None) -> V3Container:
        return container

    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    def resolve_orchestrator(_: Any = None) -> OrchestratorService:
        return service

    router = create_v3_router(resolve_container, resolve_orchestrator, {})

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    task = Task(title="Mismatch task", status="in_progress", pending_gate="before_plan")
    container.tasks.upsert(task)

    resp = client.post(f"/api/v3/tasks/{task.id}/approve-gate", json={"gate": "before_commit"})
    assert resp.status_code == 400
    assert "mismatch" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Backward compat: approval_mode mapping
# ---------------------------------------------------------------------------


def test_backward_compat_approval_mode_maps(tmp_path: Path) -> None:
    """auto_approve maps to autopilot; human_review maps to review_only when hitl_mode absent."""
    auto = Task.from_dict({"title": "Auto", "approval_mode": "auto_approve"})
    assert auto.hitl_mode == "autopilot"

    human = Task.from_dict({"title": "Human", "approval_mode": "human_review"})
    assert human.hitl_mode == "review_only"


def test_explicit_hitl_mode_preserved() -> None:
    """When hitl_mode is explicitly set in data, it overrides the mapping."""
    data = {"title": "Explicit", "hitl_mode": "supervised", "approval_mode": "auto_approve"}
    task = Task.from_dict(data)
    assert task.hitl_mode == "supervised"


# ---------------------------------------------------------------------------
# Modes endpoint uses modes.py
# ---------------------------------------------------------------------------


def test_modes_endpoint_uses_modes_py(tmp_path: Path) -> None:
    """GET /collaboration/modes returns data from MODE_CONFIGS."""
    from fastapi.testclient import TestClient

    from feature_prd_runner.collaboration.modes import MODE_CONFIGS
    from feature_prd_runner.v3.api.router import create_v3_router

    container = V3Container(tmp_path)

    def resolve_container(_: Any = None) -> V3Container:
        return container

    bus = EventBus(container.events, container.project_id)
    service = OrchestratorService(container, bus)

    def resolve_orchestrator(_: Any = None) -> OrchestratorService:
        return service

    router = create_v3_router(resolve_container, resolve_orchestrator, {})

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/api/v3/collaboration/modes")
    assert resp.status_code == 200
    data = resp.json()
    modes = data["modes"]
    assert len(modes) == len(MODE_CONFIGS)
    mode_names = [m["mode"] for m in modes]
    assert "autopilot" in mode_names
    assert "supervised" in mode_names
    assert "collaborative" in mode_names
    assert "review_only" in mode_names

    # Verify descriptions come from modes.py, not hardcoded
    supervised = next(m for m in modes if m["mode"] == "supervised")
    assert supervised["description"] == MODE_CONFIGS["supervised"].description
